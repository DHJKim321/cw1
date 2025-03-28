import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel
import pandas as pd
import regex as re
import os
import tqdm
from concurrent.futures import Future
import time
import queue
import threading
from modules.args_extractor import get_args

args = get_args()

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'movies.csv'))
EMBEDDING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'embeddings.npy'))
MAX_BATCH_SIZE = args.batch_size
MAX_WAITING_TIME = args.max_waiting_time
use_queue_batching = args.use_queue_batching

class QueryRequest(BaseModel):
    query: str
    k: int = 2

# Device selection
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

#------------------Local START--------------------
# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
#------------------Local END----------------------

# #------------------Cluster START------------------
# # 1. Load embedding model
# LOCAL_MODEL_PATH = "/home/s1808795/.cache/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/84344a23ee1820ac951bc365f1e91d094a911763"
# embed_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
# embed_model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
# # Basic Chat LLM
# LOCAL_CHAT_MODEL_PATH = "/home/s1808795/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
# chat_tokenizer = AutoTokenizer.from_pretrained(LOCAL_CHAT_MODEL_PATH, local_files_only=True)
# chat_model = AutoModelForCausalLM.from_pretrained(LOCAL_CHAT_MODEL_PATH, local_files_only=True)
# chat_pipeline = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer)
# #------------------Cluster END--------------------

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def get_embedding_batch(texts: list[str]) -> np.ndarray:
    inputs = embed_tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
if os.path.exists(EMBEDDING_PATH):
    print("Existing embeddings found. Loading...")
    doc_embeddings = np.load(EMBEDDING_PATH)
else:
    print("Embeddings not found. Computing...")
    doc_embeddings = []
    for doc in tqdm.tqdm(documents):
        doc_embeddings.append(get_embedding(doc))
    doc_embeddings = np.vstack(doc_embeddings)
    np.save(EMBEDDING_PATH, doc_embeddings)

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int) -> str:
    start = time.time()
    query_emb = get_embedding(query)
    retrieved_docs = retrieve_top_k(query_emb, k)

    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n"

    generated_text = chat_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]
    print(f"[RAG] Total processing time: {time.time() - start:.2f}s")
    answer_start = generated_text.find("Answer:")
    if answer_start != -1:
        generated_text = generated_text[answer_start + len("Answer:"):].strip() + "..."
    return generated_text

def rag_pipeline_batch(queries: list[str], k: int) -> list[str]:
    start = time.time()

    query_embs = get_embedding_batch(queries)

    retrieved_docs = [retrieve_top_k(query_emb, k) for query_emb in query_embs]
    contexts = ["\n".join(docs) for docs in retrieved_docs]

    prompts = [
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n"
        for query, context in zip(queries, contexts)
    ]

    try:
        raw_outputs = chat_pipeline(prompts, max_new_tokens=50, do_sample=True)
    except Exception as e:
        print(f"[RAG] Batch generation failed, falling back to sequential. Error: {e}")
        raw_outputs = [
            chat_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]
            for prompt in prompts
        ]

    answers = []
    for output in raw_outputs:
        text = output[0]["generated_text"]
        answer_start = text.find("Answer:")
        if answer_start != -1:
            text = text[answer_start + len("Answer:"):].strip()
        answers.append(text + "...")

    print(f"[RAG] Total processing time for batch: {time.time() - start:.2f}s")
    return answers

def process_requests_batch():
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

            results = rag_pipeline_batch(queries, ks[0])  

            for future, result in zip(futures, results):
                future.set_result(result)

# Start the background thread
if use_queue_batching:
    print("[RAG] Starting background thread for request processing...")
    threading.Thread(target=process_requests_batch, daemon=True).start()

@app.post("/rag")
def predict(payload: QueryRequest, background_tasks: BackgroundTasks):
    if use_queue_batching:
        future = Future()
        request_queue.put({"payload": payload, "future": future})
        print(f"[RAG] Added request to queue: {payload.query}")
        # Add the future to the background tasks
        async def wait_for_future(future: Future):
            future.result()
        
        background_tasks.add_task(wait_for_future, future)
        return {"query": payload.query, "result": future.result()}
    else:
        print(f"[RAG] Processing request directly: {payload.query}")
        result = rag_pipeline(payload.query, payload.k)
        return {"query": payload.query, "result": result}

if __name__ == "__main__":
    print("[RAG] Starting RAG service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
