# Submission — Day 18 Lakehouse Lab (Track 2)

Đường chạy: **lightweight** (`deltalake` 1.6.2 + `pyiceberg` 0.11.1 + DuckDB 1.5.5 +
Polars 1.43.2), Python 3.11.9 / Windows 11. Không Docker, không JVM, không network.

## Cổng chấm

| Gate | Kết quả |
|---|---|
| `make smoke` | 9/9 check |
| `make test`  | **24/24 pass** (README ghi 22 — suite đã tăng thêm 2 test isolation) |
| `make run-all` | **8/8 PASS**, exit 0 |
| Assert trong notebook | **49 check, 0 FAIL, 0 error** |

Log thô: [`screenshots/03_gates.txt`](screenshots/03_gates.txt).

## Nội dung

* [`notebooks/`](notebooks/) — 8 notebook đã chạy, **giữ nguyên output cell**.
  (Thư mục `notebooks/` ở repo root bị `.gitignore` loại `*.ipynb`, nên bản đã
  thực thi được đặt ở đây để thực sự vào PR.)
* [`REFLECTION.md`](REFLECTION.md) — Top 5 Anti-Patterns, 199 từ.
* [`screenshots/01_tree_lakehouse.txt`](screenshots/01_tree_lakehouse.txt) — `tree _lakehouse/`
* [`screenshots/02_delta_log_commit.txt`](screenshots/02_delta_log_commit.txt) — nội dung một `_delta_log/*.json`
* [`screenshots/03_gates.txt`](screenshots/03_gates.txt) — output cả 3 cổng

## Số đo theo từng deliverable

| NB | Tiêu chí | Đo được |
|---|---|---|
| 1 | `_delta_log/` JSON · bad write bị chặn · `tier` thêm được | 2 commit JSON; `Cast error: Cannot cast string 'thirty' to Int64`; `tier` vào schema qua `schema_mode="merge"` |
| 2 | speedup ≥ 3× **hoặc** files-pruned ≥ 10× | **speedup 6.9×** *và* **pruning 55×** (1/55 file chứa `user_id=4242`); file 200 → 55 |
| 3 | `history()` ≥ 5 version kể cả RESTORE | 5 version (v0 WRITE → v4 **RESTORE**); MERGE 100K trong **0.15 s**; RESTORE 0.02 s; `score<0` = **0** |
| 4 | Silver < Bronze · Gold ≥ 7 ngày × 3 model | Bronze **200,000** → Silver **190,052** (dedup bỏ 9,948); Gold **8 ngày × 3 model = 24 dòng** |
| 5 | pruning ≥ 5× · field_id bền · ≥ 2 spec | pruning **10×** (lọc trên `ts`, không phải `ts_day`); `latency_millis` giữ **field_id=4**; `spec_id` **[1, 2]** cùng tồn tại |
| 6 | 4 job + job 5 | compaction **200 → 11 (18×)**; clustering skip **90%**; vacuum thu hồi **16.1 MB**; **3/3 orphan** tìm + xoá; Iceberg **20 → 3 snapshot**; checkpoint `...099.checkpoint.parquet` + `_last_checkpoint` |
| 7 | amplification · int8 · lifecycle bug | amplification **200×** (row-group 12.5 MB cho 1 blob 64 KB); int8 **5.8× nhỏ hơn**; recall@10 **0.904**, topic fidelity **1.000**; lifecycle bug: **0 hit in-table / 8 hit external index** |
| 8 | agent_version · pin version · 4 rổ Art. 10 | Silver 2 partition (`policy-v2`, `policy-v3`); replay tại pinned v0 = **1,578 step**, khớp tuyệt đối; MCP 5 turn → **1 catalog read**; **4/4 rổ Art. 10** thành partition; **334 UNCLASSIFIED** bị loại, còn **1,666/2,000** dòng dùng được |

## Hai phát hiện trái với niềm tin phổ biến (đo được, không phải trích slide)

1. **`VACUUM` không thấy orphan chưa từng commit.** NB6: sau khi vacuum ở
   `retention_hours=0`, dry-run vẫn báo 0 file, trong khi 3 file crashed-writer
   (30 ngày tuổi) vẫn nằm trên đĩa. `deltalake` chỉ thu hồi file **đã bị
   tombstone trong log**; file chưa từng vào log thì log không biết nó tồn tại.
   Phải tự làm phép hiệu tập hợp `on-disk − referenced`.
2. **`expire_snapshots` của Iceberg không xoá file nào.** NB6: snapshot
   **20 → 3**, nhưng avro **40 → 40, deleted = 0** — và metadata còn *phình ra*
   (expiry ghi thêm `metadata.json`). Job 3 chỉ làm file trở nên
   *unreferenced*; xoá là Job 4. Chạy expiry mà không quét orphan chính là lý do
   *"đã expire mà hoá đơn S3 không giảm"*. Sweep 17 manifest list mồ côi mới
   thu hồi được 37.1 KB.

## Sửa để lab chạy được trên Windows

Ba thay đổi, đều nằm ngoài phần bài tập:

1. `scripts/run_all.py` — cổng chấm **fail 0/8 trên Windows**: notebook in UTF-8
   (`✓ → ≥`) nhưng stdout của child là pipe nên Python chọn cp1252 →
   `UnicodeEncodeError` ngay `print` đầu tiên, và lỗi thật bị che bởi
   `UnicodeDecodeError` trong reader thread của process cha. Nay ép UTF-8 cả hai
   đầu (`PYTHONIOENCODING` cho child, `encoding="utf-8"` khi decode). → 8/8.
2. `scripts/lakehouse.py` — `reset_catalog()` **im lặng không làm gì trên Windows**:
   SQLAlchemy giữ `catalog.db` mở, `rmtree(..., ignore_errors=True)` nuốt
   `WinError 32`, catalog "đã reset" vẫn còn nguyên dữ liệu cũ. Nay dispose engine
   trước khi xoá. (Đây là test `test_reset_catalog_does_not_touch_siblings` — test
   duy nhất đỏ lúc đầu.)
3. `notebooks/04_medallion.py` — thiếu hẳn khối `assert` cuối, khác 7 notebook
   còn lại, nên `make run-all` không thật sự kiểm tiêu chí của NB4. Đã bổ sung
   7 check (3 layer trên đĩa, Silver < Bronze, ≥ 7 ngày, 3 model, `cost_usd` và
   `error_rate` khác 0).
