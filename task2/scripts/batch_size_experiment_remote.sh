#!/bin/bash

# === Static parameters ===
MAX_WAITING_TIME=2
NUM_USERS=10
NUM_REQUESTS=10
TOP_K=2
USE_QUEUE_BATCHING=True
REQUEST_TYPE=gradual
TOTAL_TIME=30
VERBOSE=False
OUTPUT_DIR="batch_test"

# === Range of batch sizes to test ===
BATCH_SIZES=(1 2 4 8 16 32)

# === Loop through each batch size ===
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
  echo ""
  echo "=== Running test with BATCH_SIZE = $BATCH_SIZE ==="

  # === Cleanup function on exit or interruption ===
  cleanup() {
    echo "[Cleanup] Killing server with PID $SERVER_PID"
    kill -9 $SERVER_PID 2>/dev/null
    echo ""
  }
  trap cleanup SIGINT SIGTERM EXIT

  # === Start server ===
  python serving_rag.py \
    --use_queue_batching $USE_QUEUE_BATCHING \
    --batch_size $BATCH_SIZE \
    --max_waiting_time $MAX_WAITING_TIME \
    $( [ "$VERBOSE" = "True" ] && echo "--verbose" ) &
  SERVER_PID=$!

  # === Wait for server to be ready ===
  echo "Waiting for server to start on port 8000..."
  while ! nc -z 192.168.47.132 8000; do
  sleep 0.5
done

  echo "Server is up!"

  # === Run load test ===
  python -m modules.load_tester \
    --use_queue_batching $USE_QUEUE_BATCHING \
    --num_users $NUM_USERS \
    --num_requests $NUM_REQUESTS \
    --top_k $TOP_K \
    --request_type $REQUEST_TYPE \
    --total_time $TOTAL_TIME \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    $( [ "$VERBOSE" = "True" ] && echo "--verbose" )

  # === Cleanup server ===
  cleanup

  echo "=== Finished test with BATCH_SIZE = $BATCH_SIZE ==="
done

echo ""
echo "Test Completed"