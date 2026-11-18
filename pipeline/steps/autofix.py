"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

DROP_LINE_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_IMAGE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _is_remove_image(code):
    return code in REMOVE_IMAGE_CODES or code.startswith("DUP_IMAGE_")


def _fmt(n):
    return f"{n:.6f}"


def _clip(raw):
    """Cắt box về [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích. Ném ValueError nếu dòng hỏng."""
    cid, cx, cy, w, h = parse_yolo_line(raw)
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 1e-9 or nh <= 1e-9:
        return None
    return f"{cid} {_fmt((x1 + x2) / 2)} {_fmt((y1 + y2) / 2)} {_fmt(nw)} {_fmt(nh)}"


def _remove_image(ds, stem):
    """Xoá ảnh + nhãn. Trả về tên file ảnh đã xoá (hoặc None nếu chẳng còn gì để xoá)."""
    removed = None
    img = ds.image_path(stem)
    if img is not None and img.exists():
        removed = img.name
        img.unlink()
    lab = ds.label_path(stem)
    if lab.exists():
        removed = removed or lab.name
        lab.unlink()
    return removed


def fix(ds, issues, cfg):
    fixes = []
    errors = [i for i in issues if i.severity == ERROR]   # WARN: việc của người, không đụng

    # ---- gom theo ảnh ----
    remove_stems = set()
    missing_label = set()
    drop_lines = defaultdict(set)    # stem -> {số dòng gốc}
    clip_lines = defaultdict(set)
    rename_classes = False

    for i in errors:
        code = i.code
        if _is_remove_image(code):
            remove_stems.add(i.image)
        elif code == "MISSING_LABEL_FILE":
            missing_label.add(i.image)
        elif code == "BAD_CLASS_NAME":
            rename_classes = True
        elif code in DROP_LINE_CODES and i.line is not None:
            drop_lines[i.image].add(i.line)
        elif code == "BOX_OUT_OF_IMAGE" and i.line is not None:
            clip_lines[i.image].add(i.line)

    # ---- 1) xoá ảnh (issue dòng của ảnh đã xoá sẽ bị bỏ qua ở bước sau) ----
    for stem in sorted(remove_stems):
        name = _remove_image(ds, stem)
        if name is not None:
            fixes.append({"image": stem, "line": None, "action": "remove_image",
                          "before": name, "after": "(đã xoá ảnh và nhãn)"})

    # ---- 2) sửa dòng nhãn: mỗi file đọc một lần, ghi một lần, theo số dòng GỐC ----
    for stem in sorted(set(drop_lines) | set(clip_lines)):
        if stem in remove_stems:
            continue
        path = ds.label_path(stem)
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8", errors="replace").splitlines()
        new, changed = [], False
        for ln, raw in enumerate(old, start=1):
            if ln in drop_lines.get(stem, ()):
                fixes.append({"image": stem, "line": ln, "action": "drop_line",
                              "before": raw, "after": "(đã xoá dòng)"})
                changed = True
                continue
            if ln in clip_lines.get(stem, ()):
                try:
                    clipped = _clip(raw)
                except ValueError:
                    new.append(raw)      # dòng hỏng thì để các lỗi khác lo, không đoán
                    continue
                changed = True
                if clipped is None:
                    fixes.append({"image": stem, "line": ln, "action": "drop_line",
                                  "before": raw, "after": "(box nằm ngoài ảnh, đã xoá dòng)"})
                    continue
                fixes.append({"image": stem, "line": ln, "action": "clip_box",
                              "before": raw, "after": clipped})
                new.append(clipped)
                continue
            new.append(raw)
        if changed:
            path.write_text("\n".join(new) + ("\n" if new else ""), encoding="utf-8")

    # ---- 3) thiếu file nhãn: chỉ báo người, KHÔNG tạo file ----
    for stem in sorted(missing_label - remove_stems):
        fixes.append({"image": stem, "line": None, "action": "needs_label",
                      "before": "(không có file)", "after": "chuyển người gán"})

    # ---- 4) tên lớp trong data.yaml ----
    if rename_classes:
        ypath = ds.yaml_path
        data = {}
        if ypath.exists():
            data = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        before = ds.yaml_names()
        data["names"] = list(cfg["classes"])
        ypath.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append({"image": "data.yaml", "line": None, "action": "rename_classes",
                      "before": str(list(before.values())), "after": str(list(cfg["classes"]))})

    return fixes