"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp."""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


# ---- 2 hàm adapter: chỉnh ở đây nếu API của ds / Issue khác ----------------
def _label_files(ds):
    """Trả về iterable (stem, path) của các file nhãn."""
    if hasattr(ds, "label_files"):
        return ds.label_files()          # kỳ vọng: [(stem, Path), ...]
    return [(p.stem, p) for p in sorted(ds.labels_dir.glob("*.txt"))]


def _issue(code, severity, stem, line, msg, **extra):
    return Issue(code=code, severity=severity, stem=stem, line=line,
                 message=msg, extra=extra or None)
# -----------------------------------------------------------------------------


def run(ds, cfg):
    c = cfg["label_format"]
    overflow_tol = c["overflow_tol"]
    dup_iou = c["dup_box_iou"]
    min_box_px = c["min_box_px"]
    max_box_frac = c["max_box_frac"]
    n_cls = len(ds.classes)

    issues = []

    for stem, path in _label_files(ds):
        text = path.read_text(encoding="utf-8", errors="replace")
        size = None            # lấy kích thước ảnh lười (chỉ khi cần)
        size_loaded = False
        seen = []              # [(cls, (cx, cy, w, h))] các dòng hợp lệ trước đó

        for ln, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():            # dòng trống: bỏ qua nhưng vẫn đếm số dòng
                continue

            # 1) Định dạng
            try:
                cid, cx, cy, w, h = parse_yolo_line(raw)
            except ValueError as e:
                issues.append(_issue("BAD_FORMAT", ERROR, stem, ln,
                                     f"dòng nhãn sai định dạng: {raw.strip()!r} ({e})"))
                continue

            bad = False

            # 2) ID lớp
            if not (0 <= int(cid) < n_cls):
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

            # 4) Tràn ảnh
            overflow = max(
                0 - (cx - w / 2),      # tràn trái
                (cx + w / 2) - 1,      # tràn phải
                0 - (cy - h / 2),      # tràn trên
                (cy + h / 2) - 1,      # tràn dưới
            )
            if overflow > overflow_tol:
                issues.append(_issue("BOX_OUT_OF_IMAGE", ERROR, stem, ln,
                                     f"box tràn ảnh {overflow:.4f}", overflow=overflow))

            # 5) Box lặp: cùng lớp với dòng TRƯỚC, IoU > ngưỡng -> báo ở dòng SAU
            for pcid, pbox in seen:
                if pcid == cid and iou_xywh(pbox, box) > dup_iou:
                    issues.append(_issue("DUP_BOX", ERROR, stem, ln,
                                         "box trùng với một dòng trước đó"))
                    break
            seen.append((cid, box))

            # 6) Box quá nhỏ (px)
            if not size_loaded:
                size_loaded = True
                try:
                    size = ds.image_size(stem)
                except Exception:
                    size = None
            if size:
                img_w, img_h = size
                if min(w * img_w, h * img_h) < min_box_px:
                    issues.append(_issue("BOX_TINY", WARN, stem, ln,
                                         "cạnh ngắn của box < min_box_px"))

            # 7) Box quá lớn
            if w * h > max_box_frac:
                issues.append(_issue("BOX_HUGE", WARN, stem, ln,
                                     f"box chiếm {w * h:.2%} diện tích ảnh"))

    return issues