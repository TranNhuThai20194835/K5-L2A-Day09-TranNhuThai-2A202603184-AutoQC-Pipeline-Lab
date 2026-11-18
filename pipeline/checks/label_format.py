"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

CHECK = "label_format"


def _issue(code, severity, image, line, message, **extra):
    return Issue(check=CHECK, severity=severity, code=code, image=image,
                 message=message, line=line, extra=extra)


def run(ds, cfg):
    c = cfg["label_format"]
    overflow_tol = c["overflow_tol"]
    dup_iou = c["dup_box_iou"]
    min_box_px = c["min_box_px"]
    max_box_frac = c["max_box_frac"]
    n_cls = len(ds.classes)

    issues = []

    for label_path in ds.label_files():
        stem = label_path.stem
        lines = ds.read_label_lines(stem) or []   # [(số dòng từ 1, nội dung)], đã bỏ dòng trống
        size = None
        size_loaded = False
        seen = []                                  # [(cls, box)] các dòng hợp lệ trước đó

        for ln, raw in lines:
            # 1) Định dạng (parse_yolo_line đã chặn thiếu/thừa cột, id không nguyên, nan/inf)
            try:
                cid, cx, cy, w, h = parse_yolo_line(raw)
            except ValueError as e:
                issues.append(_issue("BAD_FORMAT", ERROR, stem, ln,
                                     f"dòng nhãn sai định dạng: {raw.strip()!r} ({e})"))
                continue

            bad = False

            # 2) ID lớp
            if not (0 <= cid < n_cls):
                issues.append(_issue("BAD_CLASS_ID", ERROR, stem, ln,
                                     f"id lớp {cid} ngoài 0..{n_cls - 1}"))
                bad = True

            # 3) Box rỗng
            if w <= 0 or h <= 0:
                issues.append(_issue("BOX_ZERO", ERROR, stem, ln,
                                     f"box có w={w}, h={h} <= 0"))
                bad = True

            if bad:
                continue

            box = (cx, cy, w, h)

            # 4) Tràn ảnh (4 mép)
            overflow = max(
                0 - (cx - w / 2),
                (cx + w / 2) - 1,
                0 - (cy - h / 2),
                (cy + h / 2) - 1,
            )
            if overflow > overflow_tol:
                issues.append(_issue("BOX_OUT_OF_IMAGE", ERROR, stem, ln,
                                     f"box tràn ảnh {overflow:.4f}", overflow=overflow))

            # 5) Box lặp: cùng lớp với dòng TRƯỚC, báo ở dòng SAU (mỗi dòng báo tối đa 1 lần)
            for pcid, pbox in seen:
                if pcid == cid and iou_xywh(pbox, box) > dup_iou:
                    issues.append(_issue("DUP_BOX", ERROR, stem, ln,
                                         "box trùng với một dòng trước đó"))
                    break
            seen.append((cid, box))

            # 6) Box quá nhỏ (px): lấy cạnh ngắn
            if not size_loaded:
                size = ds.image_size(stem)         # None nếu ảnh không mở được
                size_loaded = True
            if size:
                img_w, img_h = size
                if min(w * img_w, h * img_h) < min_box_px:
                    issues.append(_issue("BOX_TINY", WARN, stem, ln,
                                         f"cạnh ngắn của box < {min_box_px}px"))

            # 7) Box quá lớn
            if w * h > max_box_frac:
                issues.append(_issue("BOX_HUGE", WARN, stem, ln,
                                     f"box chiếm {w * h:.2%} diện tích ảnh"))

    return issues