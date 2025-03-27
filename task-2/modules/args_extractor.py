import argparse

def get_args():
    parser = argparse.ArgumentParser(description="RAG-based QA service")
    parser.add_argument("--use_queue_batching", action="store_true", help="Use batching to process requests")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing requests")
    parser.add_argument("--max_waiting_time", type=int, default=10, help="Maximum time to wait for a batch (seconds)")
    parser.add_argument("--use_auto_scaler", action="store_true", help="Use an auto-scaler to adjust the number of ports")
    parser.add_argument("--auto_scaler_alg", type=str, default="basic", help="Auto-scaler algorithm to use")
    parser.add_argument("--max_ports", type=int, default=5, help="Maximum number of ports to use")
    parser.add_argument("--use_load_balancer", action="store_true", help="Use a load balancer to distribute requests")
    parser.add_argument("--num_requests", type=int, default=50, help="Number of requests to process")
    parser.add_argument("--num_users", type=int, default=5, help="Number of concurrent users")
    return parser.parse_args()