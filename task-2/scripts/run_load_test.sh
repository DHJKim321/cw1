#!/bin/bash

BATCH_SIZE=3
MAX_WAITING_TIME=2
NUM_USERS=10
NUM_REQUESTS=10
TOP_K=2
USE_QUEUE_BATCHING=True

echo "Arguments:"
for var in BATCH_SIZE MAX_WAITING_TIME NUM_USERS NUM_REQUESTS TOP_K USE_QUEUE_BATCHING; do
  printf "  %s = %s\n" "$var" "${!var}"
done

# === Cleanup function on exit or interruption ===
cleanup() {
  echo ""
  echo "[Cleanup] Killing server with PID $SERVER_PID"
  kill -9 $SERVER_PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# === Start server ===
python serving_rag.py --use_queue_batching $USE_QUEUE_BATCHING --batch_size $BATCH_SIZE --max_waiting_time $MAX_WAITING_TIME &
SERVER_PID=$!

# === Wait for server to start ===
echo "Waiting for server to start on port 8000..."
while ! nc -z localhost 8000; do
  sleep 0.5
done
echo "Server is up!"

# === Run load test ===
python -m modules.load_tester --num_users $NUM_USERS --num_requests $NUM_REQUESTS --top_k $TOP_K