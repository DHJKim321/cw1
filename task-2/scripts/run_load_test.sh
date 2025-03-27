#!/bin/bash
python serving_rag.py --use_queue_batching --batch_size 8 --max_waiting_time 10 &
SERVER_PID=$!

echo "Waiting for server to start on port 8000..."
while ! nc -z localhost 8000; do
  sleep 0.5
done
echo "Server is up!"

python -m modules.load_tester --use_queue_batching --num_requests 20 --num_users 2

# Shut down the server process
echo "Killing server with PID $SERVER_PID"
kill -9 $SERVER_PID