import multiprocessing
import threading
import time
import uvicorn
import httpx
import asyncio
import torch
import numpy as np
import os
import queue
import regex as re
import pandas as pd
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from concurrent.futures import Future
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import argparse

# -------------------------
# Configuration class
# -------------------------
class Config:
    def __init__(self, data_path, embedding_path, max_batch_size, max_waiting_time,
                 local_model_path, local_chat_model_path, worker_port_start, load_balancer_port):
        self.data_path = data_path
        self.embedding_path = embedding_path
        self.max_batch_size = max_batch_size
        self.max_waiting_time = max_waiting_time
        self.local_model_path = local_model_path
        self.local_chat_model_path = local_chat_model_path
        self.worker_port_start = worker_port_start
        self.load_balancer_port = load_balancer_port

# -------------------------
# Utility functions
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def clean_text(text):
    return re.sub(r'\[\d+\]', '', text)

def load_context(data_path):
    df = pd.read_csv(data_path)
    documents = "Title: " + df["Title"] + " Plot: " + df["Plot"]
    documents = documents.apply(clean_text).tolist()
    return documents

def get_embedding(text, embed_tokenizer, embed_model, device):
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def retrieve_top_k(query_emb, doc_embeddings, documents, k=2):
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

# -------------------------
# Worker server code
# -------------------------
def create_worker_app(port, config: Config):
    app = FastAPI()
    app.extra = {"port": port}
    
    # Set up request queue for batching
    request_queue = queue.Queue()
    
    # Get the appropriate device
    device = get_device()
    print(f"Worker on port {port}: Using device: {device}")
    
    # RAG pipeline components: use the paths from config
    print(f"Worker on port {port}: Loading embedding model...")
    embed_tokenizer = AutoTokenizer.from_pretrained(config.local_model_path, local_files_only=True)
    embed_model = AutoModel.from_pretrained(config.local_model_path, local_files_only=True).to(device)
    
    print(f"Worker on port {port}: Loading chat model...")
    chat_tokenizer = AutoTokenizer.from_pretrained(config.local_chat_model_path, local_files_only=True)
    chat_model = AutoModelForCausalLM.from_pretrained(config.local_chat_model_path, local_files_only=True).to(device)
    chat_pipeline = pipeline("text-generation",
                             model=chat_model,
                             tokenizer=chat_tokenizer,
                             device=device if device.type != "mps" else -1)
    
    # Load documents and embeddings
    print(f"Worker on port {port}: Loading documents...")
    documents = load_context(config.data_path)
    
    if os.path.exists(config.embedding_path):
        print(f"Worker on port {port}: Loading existing embeddings...")
        doc_embeddings = np.load(config.embedding_path)
    else:
        print(f"Worker on port {port}: Computing embeddings...")
        doc_embeddings = []
        for doc in tqdm(documents):
            doc_embeddings.append(get_embedding(doc, embed_tokenizer, embed_model, device))
        doc_embeddings = np.vstack(doc_embeddings)
        # Save embeddings only for the initial worker
        if port == config.worker_port_start:
            np.save(config.embedding_path, doc_embeddings)
    
    class QueryRequest(BaseModel):
        query: str
        k: int = 2
    
    def rag_pipeline(query, k=2):
        query_emb = get_embedding(query, embed_tokenizer, embed_model, device)
        retrieved_docs = retrieve_top_k(query_emb, doc_embeddings, documents, k)
        
        context = "\n".join(retrieved_docs)
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n"
        
        generated_text = chat_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]
        answer_start = generated_text.find("Answer:")
        if answer_start != -1:
            generated_text = generated_text[answer_start + len("Answer:"):].strip()
        return generated_text
    
    def process_requests():
        while True:
            batch = []
            start_time = time.time()
            while len(batch) < config.max_batch_size and (time.time() - start_time) < config.max_waiting_time:
                try:
                    batch.append(request_queue.get(timeout=config.max_waiting_time))
                except queue.Empty:
                    break
            if batch:
                # Process batch of requests
                results = [
                    (request['payload'], request['future'],
                     rag_pipeline(request['payload'].query, request['payload'].k))
                    for request in batch
                ]
                for req, fut, result in results:
                    fut.set_result(result)
    
    threading.Thread(target=process_requests, daemon=True).start()
    
    async def wait_for_future(future: Future):
        return future.result()
    
    @app.post("/rag")
    async def predict(payload: QueryRequest, background_tasks: BackgroundTasks):
        future = Future()
        request_queue.put({"payload": payload, "future": future})
        result = await asyncio.create_task(wait_for_future(future))
        return {
            "query": payload.query,
            "result": result,
            "worker_port": port
        }
    
    return app

def run_worker(port: int, config: Config):
    app = create_worker_app(port, config)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# -------------------------
# Load Balancer Code
# -------------------------
worker_lock = threading.Lock()
worker_ports = []      # List to keep track of worker ports
worker_processes = []  # List of multiprocessing.Process objects
round_robin_index = 0

load_balancer_app = FastAPI()

request_count = 0
request_count_lock = threading.Lock()

class QueryRequest(BaseModel):
    query: str
    k: int = 2

@load_balancer_app.post("/rag")
async def proxy_rag(payload: QueryRequest):
    global round_robin_index, request_count
    with request_count_lock:
        request_count += 1
    with worker_lock:
        if not worker_ports:
            return {"error": "No available workers."}
        selected_port = worker_ports[round_robin_index % len(worker_ports)]
        round_robin_index += 1
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://0.0.0.0:{selected_port}/rag",
                json={"query": payload.query, "k": payload.k},
                timeout=60,
            )
            return response.json()
        except Exception as e:
            return {"error": str(e), "worker_port": selected_port}

# -------------------------
# Autoscaler Logic
# -------------------------
def autoscaler(min_workers, max_workers, scale_up_threshold, scale_down_threshold, check_interval, config: Config):
    global request_count, worker_ports, worker_processes
    while True:
        time.sleep(check_interval)
        with request_count_lock:
            current_count = request_count
            request_count = 0
        with worker_lock:
            num_workers = len(worker_ports)
            if current_count >= scale_up_threshold and num_workers < max_workers:
                # Compute new port for the worker.
                new_port = max(worker_ports) + 1 if worker_ports else config.worker_port_start
                if new_port == config.load_balancer_port:
                    new_port += 1
                p = multiprocessing.Process(target=run_worker, args=(new_port, config))
                p.start()
                time.sleep(1)  # small delay for startup
                worker_processes.append(p)
                worker_ports.append(new_port)
                print(f"[Autoscaler] Scaling up: Added worker on port {new_port}")
            elif current_count <= scale_down_threshold and num_workers > min_workers:
                port_to_remove = worker_ports.pop()
                proc = worker_processes.pop()
                proc.terminate()
                proc.join(timeout=5)
                print(f"[Autoscaler] Scaling down: Removed worker on port {port_to_remove}")

# -------------------------
# Main entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG load balancer and worker configuration.")
    parser.add_argument("--data_path", type=str,
                        default="/home/s1808795/task2/cw1/task-2/data/movies.csv",
                        help="Path to the movies CSV data file.")
    parser.add_argument("--embedding_path", type=str,
                        default="/home/s1808795/task2/cw1/task-2/data/embeddings.npy",
                        help="Path to the embeddings file.")
    parser.add_argument("--max_batch_size", type=int, default=16,
                        help="Maximum batch size for processing requests.")
    parser.add_argument("--max_waiting_time", type=float, default=1,
                        help="Maximum waiting time (in seconds) for batching requests.")
    parser.add_argument("--local_model_path", type=str,
                        default="/home/s1808795/.cache/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/84344a23ee1820ac951bc365f1e91d094a911763",
                        help="Path to the local embedding model.")
    parser.add_argument("--local_chat_model_path", type=str,
                        default="/home/s1808795/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6",
                        help="Path to the local chat model.")
    parser.add_argument("--worker_port_start", type=int, default=8000,
                        help="Starting port for worker processes.")
    parser.add_argument("--load_balancer_port", type=int, default=8001,
                        help="Port for the load balancer.")
    parser.add_argument("--run_load_balancer", action="store_true",
                        help="Run with load balancer enabled.")
    parser.add_argument("--no_run_load_balancer", dest="run_load_balancer", action="store_false",
                        help="Run without load balancer.")
    parser.set_defaults(run_load_balancer=True)
    parser.add_argument("--run_autoscaler", action="store_true",
                        help="Enable the autoscaler.")
    parser.add_argument("--no_run_autoscaler", dest="run_autoscaler", action="store_false",
                        help="Disable the autoscaler.")
    parser.set_defaults(run_autoscaler=True)
    parser.add_argument("--autoscaler_min_workers", type=int, default=1,
                        help="Minimum number of worker processes for the autoscaler.")
    parser.add_argument("--autoscaler_max_workers", type=int, default=3,
                        help="Maximum number of worker processes for the autoscaler.")
    parser.add_argument("--autoscaler_scale_up_threshold", type=int, default=5,
                        help="Request count threshold for scaling up.")
    parser.add_argument("--autoscaler_scale_down_threshold", type=int, default=1,
                        help="Request count threshold for scaling down.")
    parser.add_argument("--autoscaler_check_interval", type=int, default=10,
                        help="Time interval (in seconds) to check for autoscaling.")
    args = parser.parse_args()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility.
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create a configuration object
    config = Config(data_path=args.data_path,
                    embedding_path=args.embedding_path,
                    max_batch_size=args.max_batch_size,
                    max_waiting_time=args.max_waiting_time,
                    local_model_path=args.local_model_path,
                    local_chat_model_path=args.local_chat_model_path,
                    worker_port_start=args.worker_port_start,
                    load_balancer_port=args.load_balancer_port)
    
    # Start with one initial worker.
    initial_port = config.worker_port_start
    p = multiprocessing.Process(target=run_worker, args=(initial_port, config))
    p.start()
    time.sleep(5)
    
    with worker_lock:
        worker_ports.append(initial_port)
        worker_processes.append(p)
    
    # Start autoscaler thread if enabled.
    if args.run_autoscaler:
        autoscaler_thread = threading.Thread(
            target=autoscaler,
            args=(args.autoscaler_min_workers,
                  args.autoscaler_max_workers,
                  args.autoscaler_scale_up_threshold,
                  args.autoscaler_scale_down_threshold,
                  args.autoscaler_check_interval,
                  config),
            daemon=True
        )
        autoscaler_thread.start()
        print("Autoscaler started.")
    else:
        print("Autoscaler is disabled.")
    
    # Run load balancer if enabled; otherwise, run the single worker directly.
    if args.run_load_balancer:
        print(f"Starting load balancer on port {config.load_balancer_port}...")
        uvicorn.run(load_balancer_app, host="0.0.0.0", port=config.load_balancer_port, log_level="info")
    else:
        print(f"Load balancer is disabled. Running single worker on port {initial_port}...")
        uvicorn.run(create_worker_app(initial_port, config), host="0.0.0.0", port=initial_port, log_level="info")
