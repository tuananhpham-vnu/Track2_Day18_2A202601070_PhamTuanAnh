# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # PoC — Zero-downtime embedding migration + CDF-driven erasure
#
# Chứng minh hai cơ chế **khó nhất** trong `../ARCHITECTURE.md`:
#
# * **D2 / FM1** — thêm `emb_v2` vào *cùng row*, backfill theo batch, gate cứng
#   `IS NULL = 0` trước cutover, alias swap nguyên tử.
# * **FM2** — erasure request rơi vào giữa lúc rebuild index. Rebuild ngây thơ
#   (đọc snapshot cũ) làm dữ liệu đã xoá **sống lại**; đọc Change Data Feed
#   thì không.
#
# Chạy được từ clean checkout: `python submission/bonus/poc/embedding_migration.py`
# Không cần model, không cần mạng, không đụng vào `_lakehouse/` của lab.

# %%
import shutil
import sys
import tempfile
from pathlib import Path

# Console Windows mặc định cp1252 và crash ngay dấu tiếng Việt đầu tiên —
# cùng lỗi đã làm `make run-all` fail 0/8 trước khi tôi sửa `scripts/run_all.py`.
# `hasattr` là bắt buộc: dưới Jupyter, sys.stdout là ipykernel `OutStream`,
# vốn đã UTF-8 và KHÔNG có `.reconfigure()` — gọi thẳng sẽ vỡ notebook.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

ROOT = Path(tempfile.mkdtemp(prefix="poc_emb_"))
CHUNKS = str(ROOT / "chunks")
N, DIM = 5_000, 64
SUBJECT = "matter_0007"          # thân chủ sẽ rút đồng ý giữa chừng
rng = np.random.default_rng(0)


def quantize(v: np.ndarray) -> np.ndarray:
    """int8 symmetric — cùng cách lượng tử hoá đã đo ở NB7 (recall 0.904)."""
    return np.clip(np.round(v / np.abs(v).max() * 127), -127, 127).astype("int8")


def as_vec_col(m: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel(), pa.int8()), m.shape[1])


# %% [markdown]
# ## 0. Bảng `chunks` — embedding v1 nằm TRONG row, CDF bật sẵn

# %%
subjects = [f"matter_{i % 50:04d}" for i in range(N)]
emb_v1 = quantize(rng.normal(size=(N, DIM)).astype("float32"))

write_deltalake(CHUNKS, pa.table({
    "chunk_id": pa.array([f"c{i:06d}" for i in range(N)]),
    "subject_id": pa.array(subjects),
    "emb_v1": as_vec_col(emb_v1),
}), mode="overwrite", configuration={"delta.enableChangeDataFeed": "true"})

victims = sorted({c for c, s in zip([f"c{i:06d}" for i in range(N)], subjects) if s == SUBJECT})
print(f"chunks: {DeltaTable(CHUNKS).count():,} rows · v{DeltaTable(CHUNKS).version()}")
print(f"thân chủ {SUBJECT} sở hữu {len(victims)} chunk")

# %% [markdown]
# ## 1. Thêm cột `emb_v2` — schema evolution opt-in, không rewrite bảng
#
# Đây là D2: version mới là một **cột mới trong cùng row**, không phải bảng mới.

# %%
dt = DeltaTable(CHUNKS)
empty = dt.to_pyarrow_table().slice(0, 0)
widened = empty.append_column("emb_v2", pa.array([], pa.list_(pa.int8(), DIM)))
write_deltalake(dt, widened, mode="append", schema_mode="merge")

dt = DeltaTable(CHUNKS)
print("schema sau evolution:", [f.name for f in dt.schema().fields])
print(f"số row không đổi: {dt.count():,}  ← thêm cột KHÔNG rewrite dữ liệu")

# %% [markdown]
# ## 2. Backfill theo batch — bảng vẫn phục vụ v1 suốt quá trình

# %%
PROJ = rng.normal(size=(DIM, DIM)).astype("float32")   # "model v2" = phép chiếu khác
BATCH = 1_000


def backfill(lo: int, hi: int) -> None:
    ids = [f"c{i:06d}" for i in range(lo, hi)]
    src = pa.table({"chunk_id": pa.array(ids),
                    "emb_v2": as_vec_col(quantize(emb_v1[lo:hi].astype("float32") @ PROJ))})
    (DeltaTable(CHUNKS).merge(src, "t.chunk_id = s.chunk_id", source_alias="s", target_alias="t")
     .when_matched_update({"emb_v2": "s.emb_v2"}).execute())


def pending() -> int:
    """Số row chưa có emb_v2 — chính là gate của FM1.

    Dùng `.null_count`, KHÔNG phải `sum(1 for v in col if v is None)`: iterate
    một ChunkedArray trả về Arrow scalar, và scalar null không `is None`, nên
    cách viết ngây thơ luôn đếm ra 0 và gate im lặng mở toang — đúng hình dạng
    của failure mode mà gate này sinh ra để chặn.
    """
    return DeltaTable(CHUNKS).to_pyarrow_table().column("emb_v2").null_count


for lo in range(0, 3 * BATCH, BATCH):          # cố tình mới xong 3/5 batch
    backfill(lo, lo + BATCH)
print(f"backfill 3/5 batch → còn {pending():,} row chưa có emb_v2")

# %% [markdown]
# ## 3. FM1 — gate cứng chặn cutover khi backfill chưa xong
#
# Không `COALESCE(emb_v2, emb_v1)`. Trộn hai không gian vector **không báo lỗi**,
# nó chỉ trả về kết quả sai — nên gate phải là điều kiện chặn, không phải cảnh báo.

# %%
gate_blocks_early = pending() > 0
print(f"gate trước khi xong: {'CHẶN cutover' if gate_blocks_early else 'cho qua'}  ← đúng")

for lo in range(3 * BATCH, N, BATCH):
    backfill(lo, lo + BATCH)
gate_opens_after = pending() == 0
print(f"gate sau khi xong : {'cho qua' if gate_opens_after else 'CHẶN'}  ({pending()} row NULL)")

# %% [markdown]
# ## 4. FM2 — erasure rơi vào giữa lúc rebuild index
#
# 02:00 job rebuild chụp snapshot. 02:10 thân chủ rút đồng ý → `DELETE` trên
# bảng. 02:30 job build xong và swap vào. Bảng sạch; index thì không.

# %%
base_version = DeltaTable(CHUNKS).version()          # snapshot job build đọc
snapshot = DeltaTable(CHUNKS, version=base_version).to_pyarrow_table()

DeltaTable(CHUNKS).delete(f"subject_id = '{SUBJECT}'")   # erasure request
after = DeltaTable(CHUNKS)
print(f"bảng: {snapshot.num_rows:,} → {after.count():,} row (xoá {len(victims)})")

# --- cách NGÂY THƠ: build index từ snapshot, swap ---
index_naive = set(snapshot.column("chunk_id").to_pylist())
naive_hits = len(index_naive & set(victims))

# --- cách ĐÚNG: reconcile bằng CDF trước khi swap ---
index_cdf = set(snapshot.column("chunk_id").to_pylist())
cdf = DeltaTable(CHUNKS).load_cdf(starting_version=base_version + 1).read_all()
evict = [cid for cid, ct in zip(cdf.column("chunk_id").to_pylist(),
                                cdf.column("_change_type").to_pylist()) if ct == "delete"]
index_cdf -= set(evict)
cdf_hits = len(index_cdf & set(victims))

print(f"\nCDF phát {len(evict)} delete event kèm chunk_id cần evict")
print(f"  index rebuild ngây thơ  → {naive_hits} chunk đã xoá VẪN truy hồi được  ✗ vi phạm")
print(f"  index reconcile bằng CDF → {cdf_hits} chunk đã xoá truy hồi được       ✓")

# %% [markdown]
# ## 5. Cutover + drop cột cũ
#
# Alias swap là một commit vào `model_registry` — rollback = đổi lại con trỏ,
# không phải re-embed 60 M chunk.

# %%
REGISTRY = str(ROOT / "model_registry")
write_deltalake(REGISTRY, pa.table({"alias": ["live"], "column": ["emb_v1"]}), mode="overwrite")
write_deltalake(REGISTRY, pa.table({"alias": ["live"], "column": ["emb_v2"]}), mode="overwrite")
live_col = DeltaTable(REGISTRY).to_pyarrow_table().column("column")[0].as_py()

final = DeltaTable(CHUNKS).to_pyarrow_table()
write_deltalake(CHUNKS, final.drop_columns(["emb_v1"]), mode="overwrite", schema_mode="overwrite")
cols_after = [f.name for f in DeltaTable(CHUNKS).schema().fields]
print(f"alias live → {live_col}; schema sau khi drop: {cols_after}")

# %% [markdown]
# ## ✅ Pass criteria

# %%
checks = {
    "thêm cột không rewrite row":        DeltaTable(CHUNKS).count() == N - len(victims),
    "gate chặn khi backfill dở":         gate_blocks_early,
    "gate mở khi backfill xong":         gate_opens_after,
    "CDF phát đủ delete event":          len(evict) == len(victims),
    "rebuild ngây thơ làm sống lại":     naive_hits == len(victims),
    "rebuild theo CDF không sống lại":   cdf_hits == 0,
    "cutover đổi alias sang emb_v2":     live_col == "emb_v2",
    "cột cũ đã drop":                    "emb_v1" not in cols_after and "emb_v2" in cols_after,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
shutil.rmtree(ROOT, ignore_errors=True)
assert all(checks.values()), "PoC incomplete — xem các dòng FAIL ở trên"
print("\nPoC complete.")
