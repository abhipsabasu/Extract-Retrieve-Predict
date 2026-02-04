import json
import os
import torch
import faiss
from sentence_transformers import SentenceTransformer

# 1️⃣ Load dictionary from JSON file
with open("data/geonames_obscure_places.json", "r") as f:
    my_dict = json.load(f)   # should be a dict: {key: value, ...}

# 2️⃣ Load embedding model (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 3️⃣ Extract keys and compute embeddings (batched, no progress bar)
keys = list(my_dict.keys())
embeddings = model.encode(
    keys,
    batch_size=256,
    show_progress_bar=False,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

# 4️⃣ Create FAISS index (using cosine similarity via inner product)
dim = embeddings.shape[1]
# Maximize FAISS CPU threads for faster build
try:
    faiss.omp_set_num_threads(os.cpu_count() or 1)
except Exception:
    pass

index = faiss.IndexHNSWFlat(dim, 32)  # 32 = neighbors per node
index.hnsw.efConstruction = 100
index.hnsw.efSearch = 64
index.metric_type = faiss.METRIC_INNER_PRODUCT
index.add(embeddings)

# 5️⃣ Save index and key mapping
faiss.write_index(index, "data/faiss_index.bin")
with open("data/geonames_places_list.json", "w") as f:
    json.dump(keys, f)

