#!/bin/bash

BATCH_SIZE=8
MAX_WAITING_TIME=2
NUM_USERS=100
NUM_REQUESTS=10
TOP_K=2
USE_QUEUE_BATCHING=True
REQUEST_TYPE=gradual
TOTAL_TIME=30
VERBOSE=True
IS_REMOTE=True

if [ "$IS_REMOTE" = "True" ]; then
  HOST=$(hostname -I | awk '{print $1}')
else
  HOST="127.0.0.1"
fi

echo "Arguments:"
for var in BATCH_SIZE MAX_WAITING_TIME NUM_USERS NUM_REQUESTS TOP_K USE_QUEUE_BATCHING REQUEST_TYPE TOTAL_TIME VERBOSE IS_REMOTE; do
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
  $( [ "$VERBOSE" = "True" ] && echo "--verbose" ) \
  $( [ "$IS_REMOTE" = "True" ] && echo "--is_remote" ) &

SERVER_PID=$!

# === Wait for server to start ===
echo "Waiting for server to start on port 8000..."
while ! nc -z $HOST 8000; do
  sleep 0.5
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
  --batch_size $BATCH_SIZE \
  --host $HOST \
  $( [ "$IS_REMOTE" = "True" ] && echo "--is_remote" ) \
  $( [ "$VERBOSE" = "True" ] && echo "--verbose" )