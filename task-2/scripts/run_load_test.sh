#!/bin/bash

BATCH_SIZE=8
MAX_WAITING_TIME=10
NUM_USERS=2
NUM_REQUESTS=5

python serving_rag.py --batch_size $BATCH_SIZE --max_waiting_time $MAX_WAITING_TIME &
SERVER_PID=$!

echo "Waiting for server to start on port 8000..."
while ! nc -z localhost 8000; do
  sleep 0.5
done
echo "Server is up!"

python -m modules.load_tester --num_users $NUM_USERS --num_requests $NUM_REQUESTS

echo "Killing server with PID $SERVER_PID"
kill -9 $SERVER_PID