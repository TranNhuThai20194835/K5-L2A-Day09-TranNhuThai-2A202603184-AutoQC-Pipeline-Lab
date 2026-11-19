# Báo cáo Day 09 — Trần Như Thái, 2A202603184

> Xoá mọi `TODO(học viên)` và dòng gợi ý. Mỗi ý trả lời bằng **số lấy từ `submission/reports/`**, kèm mẫu số. Cả báo cáo ~1–2 trang.

## 1. Kết luận trước

Pipeline kết luận **Hold**. Vào 224 ảnh, bước nhận dữ liệu loại 21 (ảnh hỏng, quá nhỏ, trùng, rò rỉ test) nên còn 203 ảnh; bản xuất `clean_v1` là bản nháp, chưa được phát hành. Sau vòng người duyệt (mô phỏng, ngân sách 80 ảnh) vẫn còn 38/203 ảnh (18,72%) có cảnh báo, vượt ngưỡng giữ lại 10%. Nguyên nhân chính: hàng đợi 125 ảnh lớn hơn ngân sách 80 nên 45 ảnh không ai xem. Ngoài ra, máy tự sửa 35 thay đổi nhưng còn 6 lỗi bắt buộc (ảnh chưa có nhãn) phải chuyển người gán.

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT | 8 ảnh | máy | Trùng từng byte với ảnh khác, loại ảnh khỏi dataset |
| DUP_IMAGE_NEAR | 3 ảnh | máy | Trùng gần đúng (phash ≤ 8), loại ảnh; ngưỡng đã calibrate |
| LEAK_TEST | 5 ảnh | máy | Trùng ảnh kho test/val: giữ lại sẽ làm rò rỉ dữ liệu đánh giá |
| CORRUPT_IMAGE | 2 ảnh | máy | Không mở được, không dùng để huấn luyện được |
| BAD_SIZE | 3 ảnh | máy | Cạnh ngắn < 250px, loại ảnh |
| MISSING_LABEL_FILE | 6 ảnh | người | Máy không được bịa nhãn và không tạo file rỗng (rỗng nghĩa là "ảnh nền"); ghi `needs_label` |
| BAD_CLASS_ID | 6 dòng / 6 ảnh | máy | Id lớp lạ (-1, 4, 7, 93), xoá dòng |
| BAD_FORMAT | 3 dòng / 3 ảnh | máy | Có `nan` hoặc thiếu cột, xoá dòng |
| BOX_ZERO | 4 dòng / 4 ảnh | máy | Box rộng hoặc cao bằng 0, xoá dòng |
| DUP_BOX | 7 dòng / 7 ảnh | máy | Box lặp cùng lớp, xoá dòng sau, giữ dòng đầu |
| BOX_OUT_OF_IMAGE | 8 dòng / 8 ảnh | máy | Cắt box về trong [0,1], giữ nguyên lớp |
| BAD_CLASS_NAME | 2 tên lớp (Person, Bus) | máy | Ghi lại `names` trong `data.yaml` đúng `cfg["classes"]` |
| POSSIBLE_MISSING | 53 cờ / 52 ảnh | người | Model thấy vật mà nhãn không có box; model có thể sai nên cần người xác nhận |
| POSSIBLE_FALSE_POS | 29 dòng / 27 ảnh | người | Box nhãn không khớp vật nào model thấy; precision thấp (0,31 trên calib) nên đa số là báo động giả |
| POSSIBLE_LOOSE_BOX | 28 cờ / 24 ảnh | người | Đúng vật nhưng box lỏng/lệch; có vẽ lại hay không là quyết định của người |
| POSSIBLE_WRONG_CLASS | 18 ảnh | người | Gán nhầm lớp hợp lệ (IoU 0,77 đến 0,99): quy tắc cú pháp không bắt được |
| EMPTY_LABEL | 12 ảnh | người | File rỗng: "ảnh nền" hay "thiếu nhãn" chỉ người mới biết |
| BOX_TINY | 6 dòng / 4 ảnh | người | Cạnh ngắn < 12px, cảnh báo, máy không sửa |
| BOX_HUGE | 3 ảnh | người | Box chiếm trên 95% ảnh, cảnh báo, máy không sửa |
`05_review_queue.csv` xếp ảnh theo `priority` = tổng trọng số của các mã cảnh báo trên ảnh (khai báo trong `review.weights`), với mã của model được nhân thêm theo conf. Ảnh chưa có nhãn (`NEEDS_LABEL`, trọng số 10) lên đầu vì không ai gán thì ảnh vô dụng; tiếp theo là gán nhầm lớp (3), rồi thiếu nhãn và box thừa (2), box lỏng (1,5), cuối cùng là cảnh báo nhẹ (`BOX_TINY` 0,3). Ảnh dồn nhiều tín hiệu (như `img_0020`: 11,5) được xem trước ảnh chỉ có một tín hiệu.

Hàng đợi có 125 ảnh, ngân sách xem 80. Người duyệt xem trọn 6/6 ảnh `NEEDS_LABEL` và mọi ảnh `WRONG_CLASS`, và 74/103 ảnh có cảnh báo ngữ nghĩa (khoảng 72%). 45 ảnh còn lại không được xem, chủ yếu là box lỏng (15 ảnh) và box thừa (10 ảnh), nên gate vẫn Hold (18,72% ảnh còn cảnh báo).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 8 | DUP_IMAGE_EXACT: 1,00 / 1,00 → 1,00 / 1,00
DUP_IMAGE_NEAR: 1,00 / 0,50 → 1,00 / 0,67 | Máy tự xoá ảnh nên ưu tiên precision cao (FP = mất ảnh tốt). Bảng phẳng từ 4 đến 14 (NEAR R = 0,67, EXACT không đổi) nên chọn giữa vùng phẳng. Không chọn 16 vì sát vách sụp ở 20 (EXACT R = 0,33) mà chỉ thêm được 1 ảnh gần trùng. Trên raw loại thêm 2 ảnh (19 → 21). |
| `second_opinion.conf` | 0.25 | 0.6 | WRONG_CLASS: 0,56 / 0,83 → 0,62 / 0,83
MISSING: 0,50 / 1,00 → 0,57 / 1,00
LOOSE_BOX: 0,29 / 0,40 → 0,20 / 0,20
FALSE_POS: 0,31 / 1,00 → 0,31 / 1,00 (không phụ thuộc `conf`) | Cảnh báo cho người xem nên FN đắt hơn FP. 0,6 là mức cao nhất mà MISSING còn giữ recall 1,00 (0,7 tụt còn 0,88; 0,8 còn 0,75), WRONG_CLASS precision tăng. Đánh đổi: LOOSE_BOX xấu đi (R 0,40 → 0,20, chỉ 5 lỗi chèn). Trên raw, cảnh báo giảm 273 → 153, hàng đợi 145 → 125. |

TODO(học viên): **một chỗ calibration có thể đánh lừa bạn**, kèm bằng chứng từ dữ liệu của bạn (vd. mở xem vài cặp ảnh bị cho là trùng; số ảnh bị loại ở bước 1 có hợp lý không?).

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
(a) Gate chặn đúng.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt): vòng 3 vẫn còn **6 lỗi bắt buộc** `MISSING_LABEL_FILE` (6 ảnh chưa có file nhãn; máy không được bịa nhãn nên chỉ ghi `needs_label`, và ghi lại ở `03b_fixes_after_review.csv`). Gate kết luận **Rework**, mã thoát **20**, và pipeline in "KHÔNG xuất: còn lỗi bắt buộc" nên thư mục `work/exp_nb` không được tạo, dataset không được xuất. Trong lần chạy bình thường (ngân sách 80), 6 ảnh này nằm đầu hàng đợi (trọng số 10) nên được người xem trước, `n_errors` về 0 và gate chỉ còn Hold vì tỉ lệ cảnh báo 18,72%. Hai lần chạy khác nhau đúng một yếu tố là ngân sách người duyệt, nên đây là bằng chứng gate phản ứng đúng với lỗi bắt buộc chưa được xử lý.
(b) Điều pipeline không bắt được.
1. **Box lỏng nhẹ.** `POSSIBLE_LOOSE_BOX` chỉ cờ khi IoU giữa nhãn và dự đoán cùng lớp nhỏ hơn `iou_loose = 0,5`; box lệch ít (IoU từ 0,5 đến khoảng 0,85) không bao giờ bị cờ. Ngay cả ở mức cờ được, calibration cho recall chỉ **0,40 ở conf = 0,25 và 0,20 ở conf = 0,6** (bỏ sót 4 trên 5 lỗi chèn). Trên raw, 15 ảnh chỉ có cờ này nằm cuối hàng đợi nên không ai xem.
2. **Ảnh trùng bị biến đổi mạnh.** Trên calib, `DUP_IMAGE_NEAR` có recall tối đa **0,67** ở mọi `phash_dist` an toàn (2 trong 6 ảnh gần trùng lọt, kể cả khi ngưỡng lên 14); nâng lên 20 thì phá luôn `DUP_IMAGE_EXACT` (R = 0,33). Đây là loại ảnh bị cắt rồi thu nhỏ mà hash cảm nhận không nhận ra.
3. **Lỗi mà model lớn cũng mắc.** Nhãn trong raw được dựng từ model, và `second_opinion` cũng dùng model. Nếu nhãn sai trùng với dự đoán của model lớn (cùng lớp, IoU cao), cả `WRONG_CLASS` lẫn `LOOSE_BOX` đều im lặng. **[điền: ví dụ ảnh cụ thể bạn tìm được]**
4. **`LEAK_TEST` chưa được đo.** Calib không có ca rò rỉ test nào, nên recall của check này chưa có con số, dù trên raw nó bắt được 5 ảnh.
## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa train: ghi "chưa chạy" và lý do, rồi bỏ qua phần còn lại của mục này. Nếu đã train: dán bảng từ `submission/results/comparison.md` (model, epoch, seed, số ảnh train mỗi biến thể).

TODO(học viên): chênh mAP50 (clean − raw) so với **độ lệch chuẩn giữa các seed**; kết luận **trung thực** (nằm trong nhiễu thì nói là chưa kết luận được); ≥ 2 giả thuyết giải thích (ảnh trùng/rò rỉ test trong `raw`, nhãn máy còn lại trong `clean`, test chỉ 100 ảnh…) và cách kiểm.

## 6. Quyết định và ghi chú AI
(a) Quyết định cuối: Hold.** Không còn lỗi bắt buộc (`n_errors = 0`), nhưng 38/203 ảnh (18,72%) còn cảnh báo, vượt ngưỡng 10%. Để thành **Release** cần còn tối đa 20 ảnh có cảnh báo (20/203 ≈ 9,85%), tức xử lý thêm ít nhất 18 ảnh. Cách làm:
1. Tăng ngân sách người duyệt (hoặc thêm người) để xem phần hàng đợi chưa ai xem, hiện 45 trên 125 ảnh, ưu tiên các ảnh còn cảnh báo. Ảnh đã duyệt thì cảnh báo được coi là đã xử lý.
2. Người gán nhãn cho 6 ảnh `MISSING_LABEL_FILE` (đã nằm đầu hàng đợi) và phân loại 12 ảnh `EMPTY_LABEL` là "ảnh nền" hay "thiếu nhãn".
3. Chạy lại pipeline và kiểm tra tỉ lệ cảnh báo, không hạ `conf` hay nâng ngưỡng chỉ để đạt Release (xem mục 3).
(b) Dùng AI:** có, Claude (Anthropic) viết code 4 module (`label_format`, `gate`, `autofix`, `second_opinion`).
- **Prompt chính:** gửi file stub (docstring mô tả đầu vào/đầu ra) và file test của từng module, kèm yêu cầu "hoàn thành file theo yêu cầu"; khi lỗi thì dán nguyên traceback của `pytest`.
- **Cách kiểm code AI:** chạy `python -m pytest tests/test_<tên>.py -q` sau mỗi module, rồi cả bộ (mục tiêu 79 test); đọc lại logic với các ca bẫy trong docstring (nhiều dòng lỗi trong một file, dòng trống, một dòng 2 lỗi); mở xem kết quả trên dữ liệu thật bằng `viz_labels.py`.
- **Một lần AI sai:** lần đầu AI viết `label_format.py` khi chưa thấy `common.py`, đoán sai cách tạo `Issue` (dùng trường `stem`, thiếu tham số bắt buộc `check`). Kết quả: 13/14 test fail với `TypeError: missing 1 required positional argument: 'check'`. Sửa bằng cách đọc định nghĩa thật của `Issue` trong `common.py` rồi viết lại, sau đó 14/14 test qua.
*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
