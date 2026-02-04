import faiss
import json
import torch
import os
import argparse
from typing import List
from tqdm import tqdm
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import pandas as pd

_DEFAULT_INDEX_PATH = "data/faiss_index.bin"
_DEFAULT_KEYS_PATH = "data/geonames_obscure_places.json"
_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def _load_embedder(model_name: str):
    return SentenceTransformer(model_name, device=_get_device())


@lru_cache(maxsize=1)
def _load_index(index_path: str):
    # maximize FAISS CPU threads
    try:
        faiss.omp_set_num_threads(os.cpu_count() or 1)
    except Exception:
        pass
    index = faiss.read_index(index_path)
    # If the index supports HNSW runtime tuning, prefer a lower-latency setting
    try:
        index.hnsw.efSearch = max(32, getattr(index.hnsw, "efSearch", 64))
    except Exception:
        pass
    return index


@lru_cache(maxsize=1)
def _load_keys_and_data(keys_path: str):
    with open(keys_path, "r") as f:
        data = json.load(f)
    data_lc = {k.lower(): v for (k, v) in data.items()}
    with open('data/geonames_places_list.json', 'r') as f:
        keys = json.load(f)
    return keys, data_lc


def search_faiss(query: str,
                 top_k: int = 10,
                 index_path: str = _DEFAULT_INDEX_PATH,
                 keys_path: str = _DEFAULT_KEYS_PATH,
                 model_name: str = _DEFAULT_EMBED_MODEL) -> List[str]:
    """
    Search for the most semantically similar keys to a query using a saved FAISS index.

    Args:
        query (str): The query string.
        top_k (int): Number of top results to return.
        index_path (str): Path to saved FAISS index.
        keys_path (str): Path to saved keys JSON file.
        model_name (str): SentenceTransformer model name.

    Returns:
        List of tuples: [(key, score), ...] sorted by similarity.
    """
    if query == 'no' or type(query) is not str:
        return ['no']
    # Load FAISS index, keys, and embedding model once (cached)
    index = _load_index(index_path)
    keys, data = _load_keys_and_data(keys_path)
    model = _load_embedder(model_name)

    # Compute embedding for query
    query_emb = model.encode([query], batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

    # Search in FAISS index
    scores, indices = index.search(query_emb, top_k)
    # Prepare results
    results = [keys[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file with FAISS location search")
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--col", type=str, default="gemini-flash_faiss_loc", 
                        help="Column name to process (default: gemini-flash_faiss_loc)")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    tqdm.pandas(desc="Processing")
    df['gemini-flash_faiss'] = df[args.col].progress_apply(lambda x: search_faiss(x))
    df.to_csv(args.csv_path, index=False)