# Task 2: Retrieval-Augmented Generation (RAG) Service with Batching and Load Testing

## Motivation

Retrieval-Augmented Generation (RAG) pipelines are powerful, but inference latency and throughput can become bottlenecks under load. This project builds a FastAPI-based RAG service with configurable batching, load balancing, and auto-scaling features — enabling empirical analysis of performance under different configurations.

## Key Features

- RAG Pipeline that retrieves top-k movie plots and generates answers using a small LLM.
- Optional queue-based batching of incoming requests to improve efficiency.
- Load testing script supporting both "instant" and "gradual" user request strategies.
- Auto-scaler and load balancer modules prepared for multi-instance extensions.

## How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Start the RAG service:

    ```bash
    python serving_rag.py --use_queue_batching True --batch_size 8 --max_waiting_time 2 --is_remote
    ```

3. (On a separate Terminal) Run a load test:

    ```bash
    python -m modules.load_tester --use_queue_batching True --batch_size 8 --num_users 50 --num_requests 50 --request_type gradual --host 127.0.0.1
    ```

Alternatively, use the shell scripts in `scripts/` for common scenarios (`run_load_test_remote.sh`, etc.).

## Experiment Setup

To evaluate batching effectiveness, use the `batch_size_experiment_*.sh` scripts based on compute location (local or remote). These:

- Launch a fresh server instance per batch size
- Wait for readiness
- Perform load testing with specified user traffic
- Log latency, error rates, and throughput
- Cleanup between runs

## Analysis Strategy

We calculate the following evaluation metrics alongside others that we do not mention in our report (denoted by (X)):

- Average latency
- 95th percentile latency
- Throughput
- Total Requests (X)
- Total Errors (X)
- Total Duration (X)
- Median Latency (X)
- Max Latency (X)

Full evaluation outputs are written to the `/results` directory.

## System Components

- `serving_rag.py`: FastAPI app, request queue, batching logic, embedding + generation
- `modules/load_tester.py`: Load testing engine with instant/gradual modes
- `modules/args_extractor.py`: Centralized CLI arg handling
- `modules/question_loader.py`: Retrieves evaluation questions
- `scripts/`: Automate tests and run modes
- `data/`: Dataset files and precomputed embeddings
- `results/`: .txt result files that contain evaluation metrics and latencies of requests.