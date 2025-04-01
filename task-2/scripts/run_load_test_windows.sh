#!/bin/bash

BATCH_SIZE=8
MAX_WAITING_TIME=2
NUM_USERS=10
NUM_REQUESTS=10
TOP_K=2
USE_QUEUE_BATCHING=True
REQUEST_TYPE=gradual
TOTAL_TIME=30
VERBOSE=False

echo "Arguments:"
for var in BATCH_SIZE MAX_WAITING_TIME NUM_USERS NUM_REQUESTS TOP_K USE_QUEUE_BATCHING REQUEST_TYPE TOTAL_TIME VERBOSE; do
  printf "  %s = %s\n" "$var" "${!var}"
done

cleanup() {
  echo ""
  echo "[Cleanup] Killing server with PID $SERVER_PID"
  kill -9 $SERVER_PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

python serving_rag.py \
  --use_queue_batching $USE_QUEUE_BATCHING \
  --batch_size $BATCH_SIZE \
  --max_waiting_time $MAX_WAITING_TIME \
  $( [ "$VERBOSE" = "True" ] && echo "--verbose" ) &

SERVER_PID=$!

echo "Waiting for server to start on port 8000..."
MAX_TRIES=40
TRIES=0

until curl -s http://localhost:8000/ping > /dev/null; do
  sleep 0.5
  TRIES=$((TRIES + 1))
  if [ $TRIES -ge $MAX_TRIES ]; then
    echo "Server failed to start after $((MAX_TRIES / 2)) seconds. Exiting."
    cleanup
  fi
done

echo "Server is up!"

# === Run the load test ===
python -m modules.load_tester \
  --use_queue_batching $USE_QUEUE_BATCHING \
  --num_users $NUM_USERS \
  --num_requests $NUM_REQUESTS \
  --top_k $TOP_K \
  --request_type $REQUEST_TYPE \
  --total_time $TOTAL_TIME \
  $( [ "$VERBOSE" = "True" ] && echo "--verbose" )