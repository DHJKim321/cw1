import requests
import time
import concurrent.futures
import numpy as np
from modules.args_extractor import get_args
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

URL = "http://127.0.0.1:8000/rag"

def send_request(payload):
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=10)
        end_time = time.time()
        latency = round(end_time - start_time, 3)
        return response.status_code, latency
    except requests.exceptions.RequestException:
        return 0, None

def run_load_test(users=10, num_requests=50):
    print(f"Starting load test with {users} users and {num_requests} requests each.")

    latencies = []
    errors = 0

    start_overall = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=users) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]

        for future in concurrent.futures.as_completed(futures):
            status, latency = future.result()
            if status != 200 or latency is None:
                errors += 1
            else:
                latencies.append(latency)

    end_overall = time.time()
    test_duration = end_overall - start_overall

    if latencies:
        latencies_np = np.array(latencies)
        print("\n--- Load Test Results ---")
        print(f"Total Requests: {num_requests*users}")
        print(f"Total Errors: {errors} ({(errors/(num_requests*users))*100:.2f}%)")
        print(f"Total Duration: {test_duration:.2f}s")
        print(f"Throughput: {(num_requests*users) / test_duration:.2f} requests/sec")
        print(f"Average Latency: {latencies_np.mean():.3f}s")
        print(f"Median Latency: {np.median(latencies_np):.3f}s")
        print(f"95th Percentile Latency: {np.percentile(latencies_np, 95):.3f}s")
        print(f"Max Latency: {latencies_np.max():.3f}s")
    else:
        print("No successful responses.")

if __name__ == "__main__":
    print("Starting load tester...")
    payload = {"query": "Tell me about a movie with time travel", "k": 3}
    print(f"Sample payload: {payload}")
    args = get_args()
    if args.use_queue_batching:
        print("Using batching to process requests")
    if args.use_auto_scaler:
        print("Using an auto-scaler to adjust the number of ports")
    if args.use_load_balancer:
        print("Using a load balancer to distribute requests")
    num_requests = args.num_requests
    num_users = args.num_users
    run_load_test(users=num_users, num_requests=num_requests)
