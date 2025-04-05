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

args = get_args()

# Logging function
def log(msg):
    if args.verbose:
        print(msg)

URL = f"http://{args.host}:7999/rag"

def get_payloads(total_requests, k):
    """
    Generate payloads for the load test.
    Each payload contains a random question and the number of top results to return.
    The dataset comes from the question_loader module.

    Input:
        total_requests: Total number of requests to generate.
        k: Number of top results to return.
    Output:
        payloads: List of payloads for the load test.
    """
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
    """
    Send a request to the server with the given payload.
    Measure the latency of the request and return the response.

    Input:
        payload: The payload to send in the request.
    Output:
        response: The response from the server.
        status_code: The status code of the response (e.g. 200 OK, 404 NOT FOUND, 500 INTERNAL SERVER ERROR).
        latency: The latency of the request in seconds.
    """
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload)
        end_time = time.time()
        latency = round(end_time - start_time, 3)
        return response.json().get("result"), response.status_code, latency
    except Exception as e:
        log(f"Unexpected error in send_request: {e}")
        return None, 0, None
    
def print_save_results(users, num_requests, latencies, errors, duration):
    """
    Print and save the results of the load test.

    Input:
        users: Number of users in the load test.
        num_requests: Number of requests per user.
        latencies: List of latencies for each request.
        errors: Number of errors encountered during the load test.
        duration: Total duration of the load test in seconds.
    """
    total_requests = users * num_requests
    if latencies:
        latencies_np = np.array(latencies)
        log(f"\n--- Load Test Results @ {time.strftime('%X')} ---")
        log(f"Total Requests: {total_requests}")
        log(f"Total Errors: {errors} ({(errors / total_requests) * 100:.2f}%)")
        log(f"Total Duration: {duration:.2f}s")
        log(f"Throughput: {total_requests / duration:.2f} requests/sec")
        log(f"Average Latency: {latencies_np.mean():.3f}s")
        log(f"Median Latency: {np.median(latencies_np):.3f}s")
        log(f"95th Percentile Latency: {np.percentile(latencies_np, 95):.3f}s")
        log(f"Max Latency: {latencies_np.max():.3f}s")

        log("Saving results...")
        output_dir = args.output_dir + "/" if args.output_dir != "" else ""
        if not os.path.exists("results/" + output_dir):
            os.makedirs("results/" + output_dir)
        file_name = f"results/{output_dir}load_test_results_use_batching={args.use_queue_batching}_batch_size={args.batch_size}_total_requests={num_requests * num_users}_k={args.top_k}_request_type={args.request_type}.txt"
        with open(file_name, "a") as f:
            f.write(f"Latencies: {latencies}\n")
            f.write(f"Total Requests: {total_requests}\n")
            f.write(f"Throughput: {total_requests / duration:.2f} requests/sec\n")
            f.write(f"Average Latency: {latencies_np.mean():.3f}s\n")
            f.write(f"95th Percentile Latency: {np.percentile(latencies_np, 95):.3f}s\n")
    else:
        log("No successful responses.")
    
def run_load_test_instant(users, num_requests, payloads):
    """
    Run the load test with all users sending requests at the same time.
    The total number of requests is users * num_requests.
    Each user sends num_requests requests in parallel.

    Input:
        users: Number of users in the load test.
        num_requests: Number of requests per user.
        payloads: List of payloads for the load test.
    Output:
        answers: List of answers from the server.
        latencies: List of latencies for each request.
        errors: Number of errors encountered during the load test.
    """
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
    print_save_results(users, num_requests, latencies, errors, duration)
    return answers, latencies, errors

def run_load_test_gradual(users, num_requests, payloads, total_time):
    """
    Run the load test with users sending requests gradually over a specified total time.
    Each user sends num_requests requests in parallel, with a delay between each request.

    Input:
        users: Number of users in the load test.
        num_requests: Number of requests per user.
        payloads: List of payloads for the load test.
        total_time: Total time to run the load test in seconds.
    Output:
        answers: List of answers from the server.
        latencies: List of latencies for each request.
        errors: Number of errors encountered during the load test.
    """
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
        # Create a thread pool for user tasks
        # Each user will send num_requests requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=users) as user_executor:
            tasks = []
            for i in range(users):
                user_payloads = payloads[i * num_requests:(i + 1) * num_requests]
                tasks.append(user_executor.submit(user_task, user_payloads))
            for task in concurrent.futures.as_completed(tasks):
                all_futures.extend(task.result())

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(all_futures):
            response, status, latency = future.result()
            if status != 200 or latency is None:
                errors += 1
            else:
                latencies.append(latency)
                answers.append(response)

    duration = time.time() - start_time
    print_save_results(users, num_requests, latencies, errors, duration)
    return answers, latencies, errors
    
def run_load_test(users, num_requests, payloads, request_type, total_time):
    """
    Run the load test based on the specified request type.
    The request type can be "instant" or "gradual".

    Input:
        users: Number of users in the load test.
        num_requests: Number of requests per user.
        payloads: List of payloads for the load test.
        request_type: Type of request to send ("instant" or "gradual").
        total_time: Total time to run the load test in seconds (only used for "gradual").
    Output:
        answers: List of answers from the server.
        latencies: List of latencies for each request.
        errors: Number of errors encountered during the load test.
    Error:
        ValueError: If the request type is unknown.
    """
    log(f"Starting load test with {users} users and {num_requests} requests each.")
    log(f"Request Mode: {request_type}")

    if request_type == "instant":
        return run_load_test_instant(users, num_requests, payloads)
    elif request_type == "gradual":
        return run_load_test_gradual(users, num_requests, payloads, total_time)
    else:
        raise ValueError(f"Unknown request_type: {request_type}")

if __name__ == "__main__":
    log("Starting load tester...")
    if args.use_queue_batching:
        log("Using batching to process requests")
    if args.use_auto_scaler:
        log("Using an auto-scaler to adjust the number of ports")
    if args.use_load_balancer:
        log("Using a load balancer to distribute requests")
    num_requests = args.num_requests
    num_users = args.num_users
    top_k = args.top_k
    request_type = args.request_type
    total_time = args.total_time

    log("Loading test questions...")
    question_loader = modules.question_loader.QuestionLoader(is_remote=args.is_remote)
    questions = question_loader.load_questions()
    log(f"Loaded {len(questions)} questions.")

    log("Generating payloads...")
    payloads = get_payloads(num_requests * num_users, top_k)
    log(f"Generated {len(payloads)} payloads.")

    answers, latencies, errors = run_load_test(num_users, num_requests, payloads, request_type, total_time)
    log("Load test completed.")