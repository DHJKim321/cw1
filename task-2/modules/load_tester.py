import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import requests
import time
import concurrent.futures

# Load balancer URL
URL = "http://127.0.0.1:8000/rag"

# Sample request payload
payload = {"query": "Tell me about a movie with time travel", "k": 3}

# Function to send a request
def send_request():
    start_time = time.time()
    response = requests.post(URL, json=payload)
    end_time = time.time()
    return response.status_code, response.json(), round(end_time - start_time, 2)

# Test function for concurrent requests
def run_load_test(concurrent_requests=10, total_requests=50):
    print(f"Starting load test with {concurrent_requests} concurrent requests and {total_requests} total requests.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request) for _ in range(total_requests)]

        for future in concurrent.futures.as_completed(futures):
            status, result, latency = future.result()
            print(f"Status: {status}, Response: {result}, Latency: {latency}s")

if __name__ == "__main__":
    run_load_test(concurrent_requests=5, total_requests=20)  # Simulate 5 users sending 20 requests
