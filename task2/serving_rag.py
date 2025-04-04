import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import regex as re
import os
import tqdm
from concurrent.futures import Future
import time
import queue
from modules.args_extractor import get_args
from fastapi.concurrency import run_in_threadpool
import threading
import sys
from transformers import AutoModelForCausalLM
from task1.task import our_knn_nearest_batch

args = get_args()

def log(msg):
    if args.verbose:
        print(msg)

def log_queue_size():
    while True:
        with open("queue_size.log", "a") as f:
            f.write(f"[Monitor] Current queue size: {request_queue.qsize()}\n")
            time.sleep(0.5)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'movies.csv'))
EMBEDDING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'embeddings.npy'))
MAX_BATCH_SIZE = args.batch_size
MAX_WAITING_TIME = args.max_waiting_time
use_queue_batching = args.use_queue_batching
is_remote = args.is_remote
CACHE_PATH = os.path.expanduser("~") + "/.cache/huggingface/hub"

class QueryRequest(BaseModel):
    query: str
    k: int = 2

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize FastAPI
app = FastAPI()
request_queue = queue.Queue()

# Load data
def clean_text(text):
    return re.sub(r'\[\d+\]', '', text)

def load_context(data_path):
    df = pd.read_csv(data_path)
    documents = "Title: " + df["Title"] + " Plot: " + df["Plot"]
    documents = documents.apply(clean_text).tolist()
    return documents

documents = load_context(DATA_PATH)

if not is_remote:
#------------------Local START--------------------
    # 1. Load embedding model
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)

    # Basic Chat LLM
    chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")
    chat_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    chat_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
#------------------Local END----------------------
else:
# #------------------Cluster START------------------
# 1. Load embedding model
    LOCAL_MODEL_PATH = f"{CACHE_PATH}/models--intfloat--multilingual-e5-large-instruct/snapshots/84344a23ee1820ac951bc365f1e91d094a911763"
    embed_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    embed_model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True).to(device)
    # Basic Chat LLM
    LOCAL_CHAT_MODEL_PATH = f"{CACHE_PATH}/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
    chat_tokenizer = AutoTokenizer.from_pretrained(LOCAL_CHAT_MODEL_PATH, local_files_only=True)
    chat_model = AutoModelForCausalLM.from_pretrained(LOCAL_CHAT_MODEL_PATH, local_files_only=True).to(device)
    chat_pipeline = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer)
# #------------------Cluster END--------------------

@torch.no_grad()
def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

@torch.no_grad()
def get_embedding_batch(texts: list[str]) -> np.ndarray:
    inputs = embed_tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
if os.path.exists(EMBEDDING_PATH):
    log("Existing embeddings found. Loading...")
    doc_embeddings = np.load(EMBEDDING_PATH)
else:
    log("Embeddings not found. Computing...")
    doc_embeddings = []
    for doc in tqdm.tqdm(documents):
        doc_embeddings.append(get_embedding(doc))
    doc_embeddings = np.vstack(doc_embeddings)
    np.save(EMBEDDING_PATH, doc_embeddings)

## You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

# def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
#     N, D = doc_embeddings.shape
#     indices = our_knn_nearest_batch(
#         N=N,
#         D=D,
#         A=doc_embeddings,
#         X=query_emb.squeeze(),
#         K=k,
#         batch_size=5000,
#         distance_metric="l2",
#         use_kernel=True
#     )

#     return [documents[i] for i in indices]


def rag_pipeline(query: str, k: int) -> str:
    start = time.time()
    query_emb = get_embedding(query)
    retrieved_docs = retrieve_top_k(query_emb, k)

    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n"

    generated_text = chat_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]
    log(f"[RAG] Total processing time: {time.time() - start:.2f}s")
    answer_start = generated_text.find("Answer:")
    if answer_start != -1:
        generated_text = generated_text[answer_start + len("Answer:"):].strip() + "..."
    return generated_text

def generate_batch(prompts: list[str], max_new_tokens: int = 50) -> list[str]:
    inputs = chat_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        outputs = chat_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=chat_tokenizer.eos_token_id,
        )

    decoded = chat_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def rag_pipeline_batch(queries: list[str], k: int) -> list[str]:
    start = time.time()

    # Embedding
    embedding_start = time.time()
    query_embs = get_embedding_batch(queries)
    log(f"[Batch Timing] Embedding took {time.time() - embedding_start:.2f}s")

    # Retrieval
    retrieval_start = time.time()
    retrieved_docs = [retrieve_top_k(query_emb, k) for query_emb in query_embs]
    log(f"[Batch Timing] Retrieval took {time.time() - retrieval_start:.2f}s")

    # Prepare prompts
    prompt_prep = time.time()
    contexts = ["\n".join(docs) for docs in retrieved_docs]
    prompts = [
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n"
        for query, context in zip(queries, contexts)
    ]
    log(f"[Batch Timing] Prompt prep took {time.time() - prompt_prep:.2f}s")

    # Generate answers
    generation_start = time.time()
    try:
        raw_outputs = generate_batch(prompts, max_new_tokens=50)
        log(f"[Batch Timing] Generation took {time.time() - generation_start:.2f}s")
    except Exception as e:
        sys.exit(1)

    # Extract answers
    answers = []
    for text in raw_outputs:
        answer_start = text.find("Answer:")
        if answer_start != -1:
            text = text[answer_start + len("Answer:"):].strip()
        answers.append(text + "...")


    log(f"[RAG] Total processing time for batch: {time.time() - start:.2f}s")
    return answers

def process_requests_batch():
    log("[Batch] Background thread started.")
    while True:
        batch = []
        start_time = time.time()

        while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < MAX_WAITING_TIME:
            try:
                batch.append(request_queue.get(timeout=MAX_WAITING_TIME))
            except queue.Empty:
                break

        if batch:
            queries = [req['payload'].query for req in batch]
            ks = [req['payload'].k for req in batch]
            futures = [req['future'] for req in batch]

            log(f"[Batch] Collected batch of size {len(batch)}")

            results = rag_pipeline_batch(queries, ks[0])  

            for future, result in zip(futures, results):
                future.set_result(result)
            
            log("[Batch] Completed batch and returned results.")

if use_queue_batching:
    log("[RAG] Starting background batching thread...")
    threading.Thread(target=process_requests_batch, daemon=True).start()
    threading.Thread(target=log_queue_size, daemon=True).start()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/rag")
async def predict(payload: QueryRequest):
    if use_queue_batching:
        future = Future()
        request_queue.put({"payload": payload, "future": future})
        log(f"[RAG] Added request to queue: {payload.query}")
        result = await run_in_threadpool(future.result)
        return {"query": payload.query, "result": result}
    else:
        log(f"[RAG] Processing request directly: {payload.query}")
        result = rag_pipeline(payload.query, payload.k)
        return {"query": payload.query, "result": result}

if __name__ == "__main__":
    log("[RAG] Starting RAG service...")
    uvicorn.run(app, host="0.0.0.0", port=7999)
