"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401

CHECK = "second_opinion"


def _issue(code, stem, message, line=None, **extra):
    return Issue(check=CHECK, severity=WARN, code=code, image=stem,
                 message=message, line=line, extra=extra)


def _valid_label_lines(ds, stem):
    """[(số dòng gốc, (id, cx, cy, w, h))] của các dòng parse được và có id lớp hợp lệ
    (cùng tiêu chí với ds.valid_boxes, nhưng giữ số dòng gốc)."""
    out = []
    for ln, raw in ds.read_label_lines(stem) or []:
        try:
            b = parse_yolo_line(raw)
        except ValueError:
            continue
        if 0 <= b[0] < len(ds.classes):
            out.append((ln, b))
    return out


def run(ds, cfg, preds=None):
    c = cfg["second_opinion"]
    conf_min = c["conf"]
    min_area_px = c["min_area_px"]
    fp_conf = c["fp_conf"]
    iou_missing = c["iou_missing"]
    iou_match = c["iou_match"]
    iou_loose = c["iou_loose"]

    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds:
            continue
        label_lines = ds.read_label_lines(stem)
        if label_lines is None:                    # không có file nhãn: việc của MISSING_LABEL_FILE
            continue

        L = [b for _, b in _valid_label_lines(ds, stem)]   # nhãn hợp lệ (id, cx, cy, w, h)
        size = ds.image_size(stem)                 # (W, H) hoặc None
        stem_preds = preds[stem]

        # ---- A) mỗi dự đoán mạnh -> tối đa 1 issue ----
        for r in stem_preds:
            rcls, rconf = int(r[0]), r[1]
            rbox = tuple(r[2:6])
            if rconf < conf_min:
                continue
            if size:
                W, H = size
                if (rbox[2] * W) * (rbox[3] * H) < min_area_px:
                    continue

            cands = []                              # [(iou, lớp nhãn)]
            for lab in L:
                iou = iou_xywh(rbox, lab[1:5])
                if iou >= iou_missing:
                    cands.append((iou, int(lab[0])))

            if not cands:
                issues.append(_issue("POSSIBLE_MISSING", stem,
                                     f"model thấy lớp {rcls} (conf {rconf:.2f}) mà nhãn không có box nào",
                                     conf=rconf, cls=rcls))
                continue

            same = [iou for iou, lc in cands if lc == rcls]
            if same:
                best = max(same)
                if best < iou_loose:
                    issues.append(_issue("POSSIBLE_LOOSE_BOX", stem,
                                         f"box lớp {rcls} khớp vật nhưng lỏng/lệch (IoU {best:.2f} < {iou_loose})",
                                         conf=rconf, iou=best))
                continue                            # nhãn đúng lớp đã phủ vật này

            best_other = max(iou for iou, _ in cands)
            if best_other >= iou_match:
                issues.append(_issue("POSSIBLE_WRONG_CLASS", stem,
                                     f"model thấy lớp {rcls} (conf {rconf:.2f}) nhưng nhãn ở đó là lớp khác "
                                     f"(IoU {best_other:.2f})",
                                     conf=rconf, iou=best_other))
            # còn lại: khác lớp với IoU trong [iou_missing, iou_match) -> mơ hồ, không cờ

        # ---- B) mỗi dòng nhãn không khớp vật nào model thấy -> FALSE_POS ----
        strong = [tuple(r[2:6]) for r in stem_preds if r[1] >= fp_conf]   # không lọc diện tích
        for ln, lab in _valid_label_lines(ds, stem):
            best = max((iou_xywh(pbox, lab[1:5]) for pbox in strong), default=0.0)
            if best < iou_missing:
                issues.append(_issue("POSSIBLE_FALSE_POS", stem,
                                     f"box nhãn dòng {ln} không khớp vật nào model thấy (IoU lớn nhất {best:.2f})",
                                     line=ln, iou=best))

    return issues