import argparse

def str2bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError(f"Invalid boolean string: {value}")

def get_args():
    parser = argparse.ArgumentParser(description="RAG-based QA service")
    parser.add_argument("--use_queue_batching", type=str2bool, default=False, help="Use queue-based batching for requests")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing requests")
    parser.add_argument("--max_waiting_time", type=int, default=10, help="Maximum time to wait for a batch (seconds)")
    parser.add_argument("--use_auto_scaler", type=str2bool, default=False, help="Use auto-scaling for ports")
    parser.add_argument("--auto_scaler_alg", type=str, default="basic", help="Auto-scaler algorithm to use")
    parser.add_argument("--max_ports", type=int, default=5, help="Maximum number of ports to use")
    parser.add_argument("--use_load_balancer", type=str2bool, default=False, help="Use a load balancer to distribute requests")
    parser.add_argument("--num_requests", type=int, default=50, help="Number of requests to process")
    parser.add_argument("--num_users", type=int, default=5, help="Number of concurrent users")
    parser.add_argument("--top_k", type=int, default=2, help="Number of context documents to retrieve")
    return parser.parse_args()