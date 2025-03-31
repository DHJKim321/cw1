import requests
import time
import concurrent.futures
import numpy as np
from modules.args_extractor import get_args
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import modules.question_loader
import random
random.seed(42)

URL = "http://127.0.0.1:8000/rag"

def get_payloads(total_requests, k):
    payloads = []
    for i in range(total_requests):
        question = random.choice(questions)
        payload = {
            "query": question,
            "k": k,
        }
        payloads.append(payload)
    return payloads

def send_request(payload):
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload)
        end_time = time.time()
        latency = round(end_time - start_time, 3)
        return response.json().get("result"), response.status_code, latency
    except Exception as e:
        print(f"Unexpected error in send_request: {e}")
        return None, 0, None
    
def print_results(users, num_requests, latencies, errors, duration):
    total_requests = users * num_requests
    if latencies:
        latencies_np = np.array(latencies)
        print(f"\n--- Load Test Results @ {time.strftime('%X')} ---")
        print(f"Total Requests: {total_requests}")
        print(f"Total Errors: {errors} ({(errors / total_requests) * 100:.2f}%)")
        print(f"Total Duration: {duration:.2f}s")
        print(f"Throughput: {total_requests / duration:.2f} requests/sec")
        print(f"Average Latency: {latencies_np.mean():.3f}s")
        print(f"Median Latency: {np.median(latencies_np):.3f}s")
        print(f"95th Percentile Latency: {np.percentile(latencies_np, 95):.3f}s")
        print(f"Max Latency: {latencies_np.max():.3f}s")
    else:
        print("No successful responses.")
    
def run_load_test_instant(users, num_requests, payloads):
    answers, latencies, errors = [], [], 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(send_request, payload) for payload in payloads]

        for future in concurrent.futures.as_completed(futures):
            response, status, latency = future.result()
            if status != 200 or latency is None:
                errors += 1
            else:
                latencies.append(latency)
                answers.append(response)

    duration = time.time() - start_time
    print_results(users, num_requests, latencies, errors, duration)
    return answers, latencies, errors

def run_load_test_gradual(users, num_requests, payloads, total_time):
    answers, latencies, errors = [], [], 0
    start_time = time.time()
    interval = total_time / (users * num_requests)
    noise_std = 0.1 * interval

    def user_task(user_payloads):
        user_futures = []
        for payload in user_payloads:
            noise = random.gauss(0, noise_std)
            time.sleep(max(0, interval + noise))
            future = executor.submit(send_request, payload)
            user_futures.append(future)
        return user_futures

    all_futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=users) as user_executor:
            tasks = []
            for i in range(users):
                user_payloads = payloads[i * num_requests:(i + 1) * num_requests]
                tasks.append(user_executor.submit(user_task, user_payloads))
            for task in concurrent.futures.as_completed(tasks):
                all_futures.extend(task.result())

        for future in concurrent.futures.as_completed(all_futures):
            response, status, latency = future.result()
            if status != 200 or latency is None:
                errors += 1
            else:
                latencies.append(latency)
                answers.append(response)

    duration = time.time() - start_time
    print_results(users, num_requests, latencies, errors, duration)
    return answers, latencies, errors
    
def run_load_test(users, num_requests, payloads, request_type, total_time):
    print(f"Starting load test with {users} users and {num_requests} requests each.")
    print(f"Request Mode: {request_type}")

    if request_type == "instant":
        return run_load_test_instant(users, num_requests, payloads)
    elif request_type == "gradual":
        return run_load_test_gradual(users, num_requests, payloads, total_time)
    else:
        raise ValueError(f"Unknown request_type: {request_type}")

if __name__ == "__main__":
    print("Starting load tester...")
    args = get_args()
    if args.use_queue_batching:
        print("Using batching to process requests")
    if args.use_auto_scaler:
        print("Using an auto-scaler to adjust the number of ports")
    if args.use_load_balancer:
        print("Using a load balancer to distribute requests")
    num_requests = args.num_requests
    num_users = args.num_users
    top_k = args.top_k
    request_type = args.request_type
    total_time = args.total_time

    print("Loading test questions...")
    question_loader = modules.question_loader.QuestionLoader()
    questions = question_loader.load_questions()
    print(f"Loaded {len(questions)} questions.")

    print("Generating payloads...")
    payloads = get_payloads(num_requests * num_users, top_k)
    print(f"Generated {len(payloads)} payloads.")

    run_load_test(num_users, num_requests, payloads, request_type, total_time)
    print("Load test completed.")