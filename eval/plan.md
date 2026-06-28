Giai đoạn 1 — Xác định mục tiêu và baseline
Chốt phạm vi:Tài liệu: báo cáo FPT 2025.
Loại dữ liệu: text, table, image, mixed.
Ngôn ngữ câu hỏi: tiếng Việt.

Lưu cấu hình baseline:Chunking.
Embedding model.
top_k.
Reranker.
Collection và filter metadata.

Chạy lại baseline để bảo đảm kết quả tái lập được.
Đầu ra: file cấu hình run và báo cáo baseline.
Giai đoạn 2 — Chuẩn hóa test set
Mỗi test case cần có:
Câu hỏi.
Ground-truth answer.
Trang hoặc vùng nguồn chính xác.
Loại chunk mong đợi.
Evidence/reference text.
Category và độ khó.
Metadata company/year.
Cờ đánh dấu câu không thể trả lời.
Review thủ công 20 câu hiện có, đặc biệt các câu image và mixed. Không dùng “image hint” như tiêu chí chính vì kiểm tra chuỗi khá mong manh.
Đầu ra: test set sạch, có ground truth retrieval và answer.
Giai đoạn 3 — Đánh giá retrieval
Đo theo từng câu:
Hit@K: có lấy đúng evidence không.
Recall@K: lấy được bao nhiêu evidence cần thiết.
MRR: evidence đúng xuất hiện ở rank nào.
Precision@K: tỷ lệ chunk liên quan.
Metadata accuracy: đúng page, document, chunk type.
Latency.
Chạy nhiều cấu hình:
top_k: 3, 5, 10, 20.
Vector search thuần.
Có/không reranker.
Có/không metadata filter.
So sánh text, table, image riêng.
Gate đề xuất: retrieval chưa đạt Recall@10 khoảng 85–90% thì chưa vội đánh giá generation sâu.
Giai đoạn 4 — Phân tích retrieval failures
Phân loại từng lỗi:
Không ingest đúng evidence.
Caption/OCR mất thông tin.
Bounding box hoặc page metadata sai.
Chunking làm vỡ evidence.
Query không khớp embedding.
Evidence có trong candidates nhưng ranking thấp.
Ground truth/test case sai.
Với 7 failure hiện tại, kiểm tra theo thứ tự:
Evidence có tồn tại trong Qdrant không?
Metadata page/type có đúng không?
Có vào top 20 nhưng rớt khỏi top 10 không?
Caption ảnh có chứa thông tin cần trả lời không?
Đầu ra: failure report có nguyên nhân, không chỉ pass/fail.
Giai đoạn 5 — Đánh giá generation
Khi retrieval đủ ổn, đánh giá câu trả lời cuối:
Correctness.
Faithfulness: mọi khẳng định phải được context hỗ trợ.
Answer relevance.
Citation/page accuracy.
Completeness.
Refusal đúng với câu không có đáp án.
Latency và token usage.
Nên có hai cách chấm:
Rule-based cho số, đơn vị, tên thực thể.
LLM judge với rubric rõ ràng.
Review thủ công một mẫu để kiểm tra judge.
Giai đoạn 6 — End-to-end và regression
Tạo một lệnh chạy toàn bộ evaluation, lưu kèm:
Thời gian chạy.
Git commit.
Cấu hình retrieval.
Model/version.
Metric tổng và theo category.
Danh sách failure.
So sánh với baseline.
Mỗi thay đổi chỉ được xem là cải thiện khi:
Metric mục tiêu tăng.
Không làm category khác giảm đáng kể.
Kết quả đã được chạy lại và lưu report.
Thứ tự thực hiện ngay
Audit và sửa schema của 20 test cases.
Nâng script retrieval eval sang Hit@K, Recall@K, MRR.
Điều tra 7 failure hiện tại và gắn nguyên nhân.
Chạy baseline mới.
Sau đó mới xây generation evaluation.

Chỉ số đánh giá: 
- Hit@5: Kiểm tra evidence đúng có xuất hiện trong 5 kết quả đầu không
- Hit@10: Giống Hit@5, nhưng kiểm tra trong 10 kết quả đầu.
- MRR: đo evidence đúng được xếp hạng cao đến mức nào (trung bình 70c)
- Page hit: Kiểm tra retrieval có lấy được chunk từ đúng trang nguồn không.
- Chunk-type hit: Kiểm tra retrieval có lấy đúng loại dữ liệu không.
  
Page hit và Chunk-type hit dùng để chẩn đoán.
Hit@5, Hit@10 và MRR mới là chỉ số retrieval chính.

Mục tiêu baseline: 
Hit@5  ≥ 75%
Hit@10 ≥ 85%
MRR    ≥ 0.60