# Multimodal RAG trên 10 triệu document pháp lý

**Bonus Challenge — Topic D** · Phạm Tuấn Anh (23020010) · Day 18 Track 2

> Chọn D vì REFLECTION của tôi kết luận anti-pattern nguy hiểm nhất là *coi
> derived index là system-of-record* — và D là bài toán duy nhất trong danh
> sách mà lỗi đó vừa **đắt** vừa **phạm luật**. Mọi số gắn thẻ `[NBx]` là số
> **đo được trên máy tôi** trong lab, không phải trích slide.

---

## 1. Problem statement

Một văn phòng luật Việt Nam cần RAG trên **10 triệu PDF** (bản án, hợp đồng,
văn bản QPPL) — hỗn hợp text số hoá và **ảnh scan + bảng**. Chunking cho ra
**~60 triệu chunk / 30 tỉ token** (30e9 ÷ ~500 token hữu dụng mỗi chunk).

Bốn ràng buộc, mỗi cái loại bỏ một kiến trúc "hiển nhiên":

1. **Search p95 < 200 ms.** Brute-force chết ở đây — [NB7] đo 2.000 vector
   hết ~1 ms, ngoại suy tuyến tính 60 M vector ≈ **30 giây**. Bắt buộc có ANN
   index. Nhưng index đó là **derived**, không phải nguồn sự thật.
2. **Embedding sẽ regenerate ≥ 2 lần.** Migration phải **zero-downtime**:
   không được có phút nào truy vấn so sánh vector v2 với v3 — hai không gian
   khác nhau, cosine giữa chúng là rác *nhưng không báo lỗi*.
3. **Reproducible sau 5 năm.** Khi một bản án trích dẫn document version cụ
   thể, retrieval phải tái lập **y hệt** — kể cả khi model và chunker đã đổi.
4. **Dữ liệu ở lại Việt Nam** (NĐ 13/2023 + Luật BVDLCN 91/2025). Hồ sơ thân
   chủ là dữ liệu cá nhân nhạy cảm; quyền xoá phải propagate tới **mọi** bản
   sao, kể cả ANN index.

*(198 từ)*

---

## 2. Architecture

```
                         ┌─ INGEST ────────────────────────────────────────────────┐
 10M PDF ──WORM upload──▶│ Bronze  pdf_raw/          15 TB  Object Lock, bất biến  │
 (15 TB)                 │         page_images/     3.6 TB  pointer, KHÔNG inline  │
                         │           │                        [NB7: inline = 200×  │
                         │           ▼                         read amplification] │
                         │ OCR + layout (self-host GPU, version-pinned)            │
                         └───────────┼─────────────────────────────────────────────┘
                                     ▼
   ┌─ SILVER (Delta, system-of-record) ──────────────────────────────────────────┐
   │ doc_versions   append-only, KHÔNG BAO GIỜ vacuum   ~150 GB                  │
   │   doc_id · doc_version · sha256(bytes) · ocr_version · chunker_version      │
   │                                                                             │
   │ chunks   partition (doc_type, year) · cluster by doc_id      [NB2: 55×]     │
   │   chunk_id = sha256(doc_id‖doc_version‖ordinal‖text)   ← content-addressed  │
   │   text · bbox · subject_id · matter_id                                      │
   │   emb_v2 INT8[1024]   ← đang phục vụ        61 GB  ┐ hai version SỐNG       │
   │   emb_v3 INT8[1024]   ← đang backfill       61 GB  ┘ CÙNG MỘT ROW           │
   │   emb_v3_f32 FLOAT[1024]  ← chỉ để rerank  245 GB                           │
   │   delta.enableChangeDataFeed = true                            [NB7]        │
   └───────────────┬──────────────────────────────────────┬──────────────────────┘
                   │ CDF: insert/update/**delete**        │ pin version
                   ▼                                      ▼
   ┌─ DERIVED (rebuildable) ─────────┐   ┌─ GOLD / GOVERNANCE ──────────────────┐
   │ HNSW int8, shard ×2, replica ×2 │   │ retrieval_manifest  (WORM)           │
   │   76 GB RAM · alias `live`      │   │   citation_id → (chunk_id, table_ver,│
   │   ← subscribe delete events     │   │    model_id, index_build_id, ts)     │
   │   ← KHÔNG phải nguồn sự thật    │   │ cited_chunks  bản sao bất biến  [FM3]│
   └──────────┬──────────────────────┘   │ pii_access_log  mọi lần đọc          │
              │                          └──────────────────────────────────────┘
              ▼
   QUERY: ANN top-100 (int8) ──▶ fetch f32 từ Silver theo chunk_id ──▶ rerank
          ~40 ms                    ~15 ms (100 point-lookup)          ~25 ms
                                                        ────────────────────────
                                                        p95 ≈ 80 ms  ✔ < 200 ms
   Catalog: Apache Polaris (REST, self-host tại VN) ── Spark · Trino · DuckDB
```

---

## 3. Bảy quyết định chính, kèm alternatives đã loại

### D1 — Embedding sống **trong row**; ANN index là derived, rebuildable

**Chọn:** cột `emb_*` nằm cùng row với `text`, `subject_id`, `matter_id`.

* **Loại: vector DB làm system-of-record.** [NB7] tôi tái hiện đúng lỗi này:
  sau khi xoá `user_042`, **0 hit trong bảng nhưng 8 hit trong external
  index**. Sync một chiều không mang theo `DELETE`, nên nội dung đã xoá vẫn
  vào được prompt RAG — vĩnh viễn, không phải "đến lần sync sau".
* **Loại: bảng `embeddings` riêng, join theo `chunk_id`.** Tái tạo đúng
  lifecycle skew đó ở quy mô nhỏ hơn, đổi lại **không được gì** — khoá join
  chính là row đó.
* **Lợi ích kèm theo:** "tìm tương tự **trong các hồ sơ tôi được phép đọc**"
  là một query duy nhất, vì vector và cột governance cùng row — [NB7] đã chạy
  đúng dạng đó với `consent_train AND license <> 'unknown'`.

### D2 — Hai embedding version **cùng một row**, không phải hai bảng

**Chọn:** thêm cột `emb_v3` bằng `schema_mode="merge"` [NB1], backfill theo
batch, cutover bằng đổi con trỏ `model_registry`, rồi drop `emb_v2`.

* **Loại: bảng `embeddings_versioned` partition theo `model_id`.** Nghe sạch
  hơn nhưng **tái sinh y hệt bug D1**: erasure giờ phải xoá ở N partition.
  Cột-trong-row thì xoá row là chết cả 3 version trong một commit ACID.
* **Loại: rebuild bảng mới rồi swap tên.** Rewrite 60 M row = đắt, và mất
  `history()` — [NB3] time travel đang giữ yêu cầu #3.
* **Con số biện hộ:** giữ song song 2 version int8 tốn **61 GB = \$1,3/tháng**.
  1,3 đô/tháng mua được lifecycle nguyên tử; ai tách bảng để tiết kiệm 61 GB
  đang đánh đổi sai chiều ba bậc độ lớn.

### D3 — Delta cho bảng, **pointer** cho ảnh trang

* **Loại: inline blob.** [NB7] đo: lấy **một** frame 64 KB phải đọc nguyên
  row-group **12,5 MB → amplification 200×**. Đơn vị I/O của Parquet là row
  group, không phải row.
* **Loại: Lance cho toàn bộ.** Lance thắng đúng bài random access trên, nhưng
  (a) **không có Change Data Feed** — mà CDF là cơ chế giữ D1 và FM2 đứng
  vững; (b) Trino/Spark/DuckDB của team khác đọc Delta. Lance đúng cho
  *feature store ảnh*, không đúng cho *system-of-record có nghĩa vụ pháp lý*.
* **Loại: Iceberg.** Sát nút — [NB5] cho thấy hidden partitioning thanh lịch
  hơn (pruning 10×). Loại vì hai lý do đo được: `pyiceberg` 0.11 chưa expose
  changelog scan như `load_cdf()`; và [NB6] `expire_snapshots` **không xoá
  file nào** (snapshot 20→3 nhưng avro 40→40, `deleted=0`, metadata còn phình
  thêm) — vận hành retention 5 năm trên đó cần thêm hẳn một orphan-sweep job.
  Delta + deletion vector xoá một row mà không rewrite file 512 MB — đúng
  hình dạng của erasure request.

### D4 — int8 cho ANN, float32 chỉ để rerank top-100

* **Loại: float32 trong index.** [NB7] đo: int8 nhỏ hơn **5,8×**, recall@10
  **0,904**, topic fidelity **1,000** — 9,6% "mất" là hoán vị giữa neighbour
  *cùng chủ đề*, tức exact-ID recall đánh giá thấp chất lượng quantization cho
  RAG. Đổi lại f32 đẩy serving RAM 76 GB → ~305 GB = **+\$1.750/tháng, +70%
  hoá đơn** (§5). Rerank thu lại phần recall đó bằng 100 point-lookup ≈ 15 ms.
* **Loại: binary/1-bit.** Rẻ hơn nữa, nhưng retrieval pháp lý là
  precision-critical và tôi **chưa đo** trên corpus này. Nguyên tắc từ [NB7]:
  recall là hàm của corpus — chỉ xét lại khi có đường cong recall thật.

### D5 — Reproducibility 5 năm = content-addressing + WORM, **không phải** time travel

**Chọn:** `chunk_id = sha256(doc_id‖doc_version‖ordinal‖text)`;
`retrieval_manifest` ghi `(chunk_id, table_version, model_id, index_build_id)`
cho mỗi trích dẫn; chunk đã trích dẫn được copy sang `cited_chunks` (Object Lock).

* **Loại: "cứ dùng time travel".** Đây là bẫy tôi suýt rơi vào. [NB3] time
  travel hoạt động hoàn hảo — nhưng chỉ sống đúng bằng retention của VACUUM.
  Giữ 5 năm = không bao giờ vacuum = log và small file tích luỹ vô hạn. Time
  travel là công cụ **vận hành 90 ngày**; nghĩa vụ 5 năm phải nằm trên WORM.
* **Loại: đóng băng cả bảng vĩnh viễn.** [NB6] checkpoint giúp đọc nhanh
  nhưng không thu nhỏ lịch sử.
* **Vì sao content-addressed:** OCR/chunker đổi → text đổi → `chunk_id` đổi →
  hệ thống **ồn ào** thay vì âm thầm trả nội dung khác dưới cùng một ID (FM4).

### D6 — Partition `(doc_type, year)`, cluster theo `doc_id`

* **Loại: partition theo ngày ingest.** Không ai query "hợp đồng nạp vào hôm
  12/3"; người ta query theo loại văn bản và theo document.
* **Loại: partition theo tỉnh/thành.** 63 partition lệch nặng (HCM + HN chiếm
  phần lớn) → small-file explosion ở các tỉnh còn lại; [NB6] compaction
  200→11 file cho thấy giá của small file là thật.
* **Con số:** [NB2] clustering đúng cột lọc cho pruning **55×** và speedup
  6,9×. [NB5] nhấn thêm: chỉ đạt 10× khi lọc **trên đúng cột đã transform**.

### D7 — Catalog: Apache Polaris self-host tại VN

* **Loại: Unity Catalog** và **AWS Glue** — ràng buộc #4 loại chúng trước cả
  lock-in: hồ sơ thân chủ không rời lãnh thổ.
* **Loại: Hive Metastore** — không REST spec, không governance cấp cột.
* Polaris nói REST Catalog spec, nên Spark/Trino/DuckDB đổi catalog là **đổi
  config, không đổi code**.

---

## 4. Failure modes — hỏng gì lúc 3 giờ sáng

### FM1 — Backfill embedding chết giữa chừng, index trộn hai không gian vector

Job backfill `emb_v3` chết ở 60%. Index build chạy theo lịch, đọc
`COALESCE(emb_v3, emb_v2)`. Giờ 60% vector ở không gian v3, 40% ở v2. Cosine
giữa hai không gian **không báo lỗi** — search vẫn 200 ms, vẫn trả 10 kết quả,
chỉ là sai. Không alert nào nổ.

* **Detect:** (a) gate cứng trước build — `count(*) WHERE emb_v3 IS NULL` phải
  = 0, nếu không thì abort; không COALESCE, không fallback im lặng.
  (b) canary 200 query có gold docs mỗi 15 phút, alert khi nDCG@10 tụt > 5%.
* **Rollback:** index build là **atomic alias swap**, index v2 giữ hot 14
  ngày → trỏ `live` về v2 trong < 60 giây.
* *Gắn Day 18:* schema evolution [NB1] — vì `emb_v2`/`emb_v3` là **hai cột
  riêng** chứ không phải một cột bị ghi đè, rollback chỉ là đổi con trỏ.

### FM2 — Erasure request rơi vào giữa lúc rebuild index

02:00 job rebuild chụp snapshot. 02:10 thân chủ rút đồng ý, `DELETE` chạy trên
Silver. 02:30 job build xong và swap vào — dữ liệu đã xoá **sống lại**. Bảng
vẫn sạch, nên mọi audit query chạy trên bảng đều xanh.

* **Detect:** (a) reconciliation bắt buộc trước swap: đọc
  `load_cdf(starting_version=<version lúc build bắt đầu>)`, mọi event
  `_change_type='delete'` phải được apply — [NB7] xác nhận CDF phát **đúng 8
  delete event kèm `doc_id` cần evict**; (b) audit đêm: lấy mẫu 10 K
  `chunk_id` đã tombstone, query index, kỳ vọng **0 hit**.
* **Rollback:** apply batch delete từ CDF vào index live ngay (HNSW hỗ trợ
  mark-deleted), rồi rebuild sạch. SLA nội bộ < 15 phút.
* *Gắn Day 18:* Change Data Feed + deletion vectors. **PoC kèm theo hiện thực
  đúng failure mode này** và chứng minh rebuild ngây thơ làm 100 chunk đã xoá
  sống lại, còn rebuild reconcile bằng CDF thì 0.

### FM3 — VACUUM xoá mất file mà một trích dẫn 5 năm đang trỏ tới

Vacuum chạy retention mặc định. 4 năm sau luật sư mở lại hồ sơ,
`retrieval_manifest` trỏ `table_version=8412` — file đã bị thu hồi. Phát hiện
**tại phiên toà**, không phải trên dashboard.

* **Phòng ngừa:** vacuum có pre-flight gate đọc version nhỏ nhất còn được pin
  trong `retrieval_manifest` và từ chối chạy nếu horizon vượt qua nó. Quan
  trọng hơn: chunk đã trích dẫn được copy sang `cited_chunks` (Object Lock)
  **ngay lúc trích dẫn**, không phụ thuộc retention của bảng nóng.
* **Rollback: không có.** Byte đã xoá là đã xoá — tôi nói thẳng thay vì vẽ ra
  một quy trình khôi phục không tồn tại.
* *Bẫy đo được ở [NB6]:* `VACUUM` **không thấy** orphan chưa từng commit —
  dry-run báo 0 file trong khi 3 file crashed-writer 30 ngày tuổi vẫn nằm trên
  đĩa. Nên dọn rác là **hai job khác nhau**: vacuum (file đã tombstone) và
  orphan sweep (phép hiệu `on-disk − referenced`). Gộp hai job này là lý do
  kinh điển của *"đã expire mà hoá đơn storage không giảm"*.

### FM4 — Nâng cấp OCR âm thầm đổi ranh giới chunk

OCR lên version mới, tách bảng tốt hơn → ranh giới chunk dịch → trích dẫn cũ
trỏ vào text khác.

* **Detect:** content-addressing (D5) làm lỗi này ồn ào — nếu byte PDF không
  đổi mà > 2% `chunk_id` trong batch re-OCR là ID mới → fail job.
* **Rollback:** `ocr_version` pin theo `doc_version`; re-OCR luôn tạo
  **doc_version mới**, không mutate bản cũ. Trích dẫn cũ vẫn resolve.

---

## 5. Chi phí back-of-envelope

Đơn giá giả định: object storage tại VN **\$0,022/GB-tháng**, GPU L4
**\$0,80/h**, node RAM-optimized 64 GB **\$0,60/h** (thay bằng báo giá thật
khi có). Ràng buộc #4 kéo theo một cái giá ít ai tính: nhà cung cấp trong nước
**không có tier archive kiểu Glacier**, nên chiến lược cold data là *nén mạnh
hơn*, không phải *đổi storage class*.

| Storage (thường xuyên) | Dung lượng | Phép tính | \$/tháng |
|---|---:|---|---:|
| Bronze PDF gốc (WORM) | 15 TB | 15.360 GB × 0,022 | 338 |
| Ảnh trang (subset scan 40%) | 3,6 TB | 3.686 GB × 0,022 | 81 |
| Silver text + layout (zstd) | 150 GB | 150 × 0,022 | 3,3 |
| `emb_v2` + `emb_v3` int8 | 122 GB | 60M × 1.024 B × 2 | 2,7 |
| `emb_v3_f32` (rerank) | 245 GB | 60M × 4.096 B | 5,4 |
| Manifest + `cited_chunks` | 20 GB | 20 × 0,022 | 0,4 |
| | | **Storage** | **\$431** |

| Compute (thường xuyên) | Phép tính | \$/tháng |
|---|---|---:|
| Serving ANN (76 GB RAM: 2 shard × 2 replica) | 4 × 730 h × 0,60 | 1.752 |
| Rerank + API | ~ | 300 |
| | **Compute** | **\$2.052** |

**Tổng thường xuyên ≈ \$2.483/tháng.**

| Một lần / theo sự kiện | Phép tính | \$ |
|---|---|---:|
| OCR 24 M trang (self-host GPU) | 24M × 0,5 s ÷ 3600 = 3.333 GPU-h × 0,80 | 2.667 |
| *(đã loại)* OCR qua API thương mại | 24M ÷ 1000 × \$1,50 | 36.000 |
| **Re-embed toàn corpus, 1 lần** | 60M × 7,3 ms ÷ 3600 = 122 GPU-h × 0,80 | **98** |

Phép tính re-embed: encoder 300 M tham số, forward-only, seq 512 →
2 × 3e8 × 512 ≈ **3,1e11 FLOP/chunk**; L4 ~121 TFLOPS bf16 ở MFU 35% ≈
42 TFLOPS hiệu dụng → **7,3 ms/chunk**.

**Ba điều bảng này nói ra, đều trái trực giác:**

1. **Re-embed toàn bộ 10 triệu document tốn \$98** — 4% của một tháng vận
   hành, trong khi ràng buộc "regenerate ≥ 2 lần" nghe như dòng chi phí đáng
   sợ nhất của đề bài. Cái đắt không phải compute mà là **choreography**:
   trộn nhầm không gian vector (FM1), mất khả năng rollback, downtime. Nên D2
   chi tiền vào *migrate an toàn*, không vào *migrate rẻ*.
2. **Serving RAM = 71% hoá đơn** → D4 (int8) là quyết định *tài chính* lớn
   nhất tài liệu này, không phải lựa chọn kỹ thuật cho vui.
3. **Storage = 17%, riêng cột embedding = 0,3%.** Tranh cãi "có nên nhét
   embedding vào bảng không" tốn nhiều team hàng tuần họp; ở đây nó là
   **\$8/tháng**. Argument đúng cho D1 là lifecycle, không bao giờ là dung lượng.

---

## 6. Tuần 1 build gì

**Slice nhỏ nhất chứng minh kiến trúc đứng được — 50.000 document (0,5%):**

1. `doc_versions` → `chunks` với `chunk_id` content-addressed, CDF bật.
2. Nạp `emb_v1`, dựng HNSW int8, đặt alias `live`, đo p95 thật.
3. **Chạy trọn một migration v1 → v2**: thêm cột bằng `schema_mode="merge"`,
   backfill theo batch, gate `IS NULL = 0`, alias swap, drop cột cũ.
4. **Chạy phép thử erasure của [NB7] trên index đang live**: xoá một
   `subject_id`, propagate qua CDF, khẳng định **0 hit ở cả bảng lẫn index** —
   đảo ngược kết quả `0 / 8` tôi đo được ở lab.

Bước 3 và 4 là hai thứ **khó** và cũng là hai thứ giết dự án này nếu sai.
Chạy được ở quy mô 1/200 thì kiến trúc đứng được.

**Cố tình *không* làm tuần 1:** OCR pipeline (dùng text-layer PDF có sẵn),
trích xuất bảng, multi-tenant authz, ảnh trang — không cái nào chứng minh hay
bác bỏ được các quyết định ở §3.

---

## PoC

[`poc/embedding_migration.py`](poc/embedding_migration.py) — 100 dòng code,
chạy từ clean checkout, không cần model/mạng, không đụng `_lakehouse/` của lab:

```bash
.venv/Scripts/python submission/bonus/poc/embedding_migration.py   # 8/8 PASS
```

Hiện thực đúng bước 3 và 4 ở trên, và chứng minh rebuild ngây thơ làm
**100 chunk đã xoá sống lại** trong khi rebuild reconcile bằng CDF thì **0**.
Notebook đã thực thi (giữ output): [`poc/embedding_migration.ipynb`](poc/embedding_migration.ipynb);
log: [`poc/RUN_LOG.txt`](poc/RUN_LOG.txt).

| Check | Kết quả |
|---|---|
| Thêm cột không rewrite row | PASS |
| Gate chặn cutover khi backfill dở (2.000 row NULL) | PASS |
| Gate mở khi backfill xong (0 row NULL) | PASS |
| CDF phát đủ 100 delete event | PASS |
| Rebuild ngây thơ làm dữ liệu đã xoá sống lại (100 hit) | PASS |
| Rebuild reconcile bằng CDF: 0 hit | PASS |
| Cutover đổi alias sang `emb_v2` | PASS |
| Drop cột `emb_v1` sau cutover | PASS |
