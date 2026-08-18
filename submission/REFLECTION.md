# REFLECTION — Top 5 Lakehouse Anti-Patterns

**Anti-pattern team tôi dễ vướng nhất: coi derived index là system-of-record — vector DB sync bỏ quên `DELETE`.**

Pipeline RAG của team dựng vector index nightly từ bảng Delta bằng one-way upsert. NB7 tái hiện đúng lỗi đó trên máy tôi: sau khi xoá `user_042` khỏi lakehouse, **0 hit trong bảng nhưng external index vẫn trả về 8 doc đã xoá**. Upsert một chiều không mang theo delete, nên dữ liệu đã xoá vẫn vào được prompt RAG — vĩnh viễn, không phải đến lần sync sau. Đây không còn là data-quality bug: nó vi phạm quyền xoá (GDPR, PDPL 91/2025).

Team tôi rủi ro cao vì (1) tối ưu recall trước, lifecycle sau; (2) không ai *sở hữu* index — nó là artifact của notebook, không retention, không audit.

Cách sửa đã đo trong NB7: giữ embedding **trong cùng row**, semantic search chạy bằng SQL (`array_cosine_similarity`) — lifecycle do chính bảng ép. Nếu buộc phải tách index, subscribe **Change Data Feed**; CDF phát đúng 8 delete event kèm `doc_id` cần evict.

Á quân là small files (NB6: 200 → 11 file, 90% file được skip sau clustering). Nhưng small files chỉ *đắt* — lifecycle skew thì *phạm luật*.
