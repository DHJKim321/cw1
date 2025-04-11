import asyncio
import time
import argparse
import numpy as np
import httpx
import random
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class RAGLoadTester:
    def __init__(self, host="http://localhost:8005", endpoint="/rag"):
        """
        Initialize the RAG Load Tester for real system tests.
        """
        self.host = host
        self.endpoint = endpoint
        self.url = f"{host}{endpoint}"
        self.results = []
        self.active_workers = {}  # worker_port -> request count
        self.worker_history = []  # (timestamp, worker_count)
        self.start_time = None
        self.worker_overhead = {}  # Records first (warm-up) response time per worker

    async def send_request(self, query, k=2, client=None):
        if client is None:
            async with httpx.AsyncClient() as client:
                return await self._do_send_request(client, query, k)
        else:
            return await self._do_send_request(client, query, k)
    
    async def _do_send_request(self, client, query, k):
        start_time = time.time()
        try:
            response = await client.post(
                self.url,
                json={"query": query, "k": k},
                timeout=60.0
            )
            end_time = time.time()
            raw_response_time = end_time - start_time
            response_data = response.json()
            worker_port = response_data.get("worker_port", "unknown")
            if worker_port != "unknown":
                self.active_workers[worker_port] = self.active_workers.get(worker_port, 0) + 1
            if self.start_time is None:
                self.start_time = start_time
            # Record the first response time for each worker as its overhead.
            if worker_port not in self.worker_overhead:
                self.worker_overhead[worker_port] = raw_response_time
                adjusted_response_time = 0.0
            else:
                adjusted_response_time = max(0, raw_response_time - self.worker_overhead[worker_port])
            result = {
                "time_sent": start_time - self.start_time,
                "time_received": end_time - self.start_time,
                "response_time": raw_response_time,
                "adjusted_response_time": adjusted_response_time,
                "query": query,
                "worker_port": worker_port,
                "success": True,
                "result_length": len(response_data.get("result", "")),
                "status_code": response.status_code,
                "simulated": False
            }
            self.results.append(result)
            return result
        except Exception as e:
            end_time = time.time()
            raw_response_time = end_time - start_time
            result = {
                "time_sent": start_time - self.start_time if self.start_time else 0,
                "time_received": end_time - self.start_time if self.start_time else 0,
                "response_time": raw_response_time,
                "adjusted_response_time": raw_response_time,
                "query": query,
                "success": False,
                "error": str(e),
                "simulated": False
            }
            self.results.append(result)
            return result
    
    async def run_load_test(self, queries, concurrency=1, delay=0.0):
        self.start_time = time.time()
        self.results = []
        self.active_workers = {}
        self.worker_history = []
        self.worker_overhead = {}
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:
            async def bounded_request(query):
                async with semaphore:
                    result = await self.send_request(query, client=client)
                    timestamp = time.time() - self.start_time
                    self.worker_history.append((timestamp, len(self.active_workers)))
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return result
            tasks = [bounded_request(query) for query in queries]
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await future
        print(f"Test completed. {len(self.results)} requests processed.")
        return self.results
    
    def save_results(self, filename_prefix):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{filename_prefix}_{timestamp}.csv"
        workers_filename = f"{filename_prefix}_workers_{timestamp}.csv"
        summary_filename = f"{filename_prefix}_summary_{timestamp}.txt"
        
        visualization_data = self.generate_visualization_data()
        response_times_csv = f"{filename_prefix}_response_times_{timestamp}.csv"
        worker_count_csv = f"{filename_prefix}_worker_count_{timestamp}.csv"
        worker_distribution_csv = f"{filename_prefix}_worker_distribution_{timestamp}.csv"
        
        visualization_data["response_times_df"].to_csv(response_times_csv, index=False)
        visualization_data["worker_count_df"].to_csv(worker_count_csv, index=False)
        visualization_data["worker_distribution_df"].to_csv(worker_distribution_csv, index=False)
        
        df = pd.DataFrame(self.results)
        df.to_csv(results_filename, index=False)
        
        worker_df = pd.DataFrame(self.worker_history, columns=["timestamp", "worker_count"])
        worker_df.to_csv(workers_filename, index=False)
        
        with open(summary_filename, "w") as f:
            summary = self.generate_summary()
            f.write(summary)
        
        print(f"Real test results saved to {results_filename}")
        print(f"Worker history saved to {workers_filename}")
        print(f"Summary saved to {summary_filename}")
        print("Visualization data saved to:")
        print(f"  - Response times: {response_times_csv}")
        print(f"  - Worker count: {worker_count_csv}")
        print(f"  - Worker distribution: {worker_distribution_csv}")
        
        return {
            "results": results_filename,
            "workers": workers_filename,
            "summary": summary_filename,
            "visualization": {
                "response_times": response_times_csv,
                "worker_count": worker_count_csv,
                "worker_distribution": worker_distribution_csv
            }
        }
    
    def generate_visualization_data(self):
        result = {}
        if self.results:
            df = pd.DataFrame(self.results)
            successful = df[df["success"] == True].copy()
            if not successful.empty:
                if len(successful) > 1:
                    z = np.polyfit(successful["time_sent"], successful["response_time"], 1)
                    p = np.poly1d(z)
                    successful["trend_line"] = p(successful["time_sent"])
                else:
                    successful["trend_line"] = successful["response_time"]
                result["response_times_df"] = successful[["time_sent", "response_time", "adjusted_response_time", "trend_line"]]
            else:
                result["response_times_df"] = pd.DataFrame(columns=["time_sent", "response_time", "adjusted_response_time", "trend_line"])
        else:
            result["response_times_df"] = pd.DataFrame(columns=["time_sent", "response_time", "adjusted_response_time", "trend_line"])
        if self.worker_history:
            result["worker_count_df"] = pd.DataFrame(self.worker_history, columns=["timestamp", "worker_count"])
        else:
            result["worker_count_df"] = pd.DataFrame(columns=["timestamp", "worker_count"])
        if self.active_workers:
            worker_distribution = pd.DataFrame([
                {"worker_port": port, "request_count": count}
                for port, count in self.active_workers.items()
            ])
            result["worker_distribution_df"] = worker_distribution
        else:
            result["worker_distribution_df"] = pd.DataFrame(columns=["worker_port", "request_count"])
        return result
    
    def generate_summary(self):
        if not self.results:
            return "No results available."
        df = pd.DataFrame(self.results)
        success_rate = df["success"].mean() * 100
        successful_requests = df[df["success"] == True]
        if len(successful_requests) > 0:
            mean_response_time = successful_requests["response_time"].mean()
            median_response_time = successful_requests["response_time"].median()
            p95_response_time = successful_requests["response_time"].quantile(0.95)
            min_response_time = successful_requests["response_time"].min()
            max_response_time = successful_requests["response_time"].max()
        else:
            mean_response_time = median_response_time = p95_response_time = min_response_time = max_response_time = "N/A"
        if self.start_time is not None:
            total_duration = df["time_received"].max()
            throughput = len(df) / total_duration if total_duration > 0 else 0
        else:
            throughput = 0
        worker_counts = {port: count for port, count in self.active_workers.items()}
        summary = [
            "# RAG Load Test Summary - Real System",
            f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Requests: {len(self.results)}",
            f"Success Rate: {success_rate:.2f}%",
            f"Mean Response Time: {mean_response_time:.4f} seconds" if isinstance(mean_response_time, float) else f"Mean Response Time: {mean_response_time}",
            f"Median Response Time: {median_response_time:.4f} seconds" if isinstance(median_response_time, float) else f"Median Response Time: {median_response_time}",
            f"95th Percentile Response Time: {p95_response_time:.4f} seconds" if isinstance(p95_response_time, float) else f"95th Percentile Response Time: {p95_response_time}",
            f"Min Response Time: {min_response_time:.4f} seconds" if isinstance(min_response_time, float) else f"Min Response Time: {min_response_time}",
            f"Max Response Time: {max_response_time:.4f} seconds" if isinstance(max_response_time, float) else f"Max Response Time: {max_response_time}",
            f"Average Throughput: {throughput:.2f} requests/second",
            "\nWorker Distribution:",
        ]
        for port, count in sorted(worker_counts.items()):
            summary.append(f"  Worker {port}: {count} requests ({count/len(self.results)*100:.1f}%)")
        return "\n".join(summary)

def load_queries_from_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def generate_sample_queries(n=10):
    queries = [
        "What is the plot of The Matrix?",
        "Tell me about Star Wars",
        "Describe the movie Inception",
        "What happens in The Godfather?",
        "Summarize the plot of Titanic",
        "What is Fight Club about?",
        "Tell me the story of The Shawshank Redemption?",
        "What's the plot of Pulp Fiction?",
        "Describe Forrest Gump",
        "What's the storyline of The Dark Knight?",
        "Tell me about The Lord of the Rings",
        "What happens in Jurassic Park?",
        "Describe Avengers: Endgame",
        "What is the plot of The Lion King?",
        "Tell me about The Silence of the Lambs",
        "What's the story of Back to the Future?",
        "Describe Avatar",
        "What happens in The Terminator?",
        "Tell me about Gladiator",
        "What's the plot of The Departed?"
    ]
    return [random.choice(queries) for _ in range(n)]

async def run_real_test(args, queries):
    print("\n=== Running test with the real system (autoscaler/load balancer state controlled externally) ===")
    tester = RAGLoadTester(host=args.host, endpoint=args.endpoint)
    await tester.run_load_test(queries, concurrency=args.concurrency, delay=args.delay)
    tester.save_results(args.output)
    print(tester.generate_summary())
    return tester.results

async def main():
    parser = argparse.ArgumentParser(description="Load test for RAG service on real system")
    parser.add_argument("--host", default="http://localhost:8001", help="Host address")
    parser.add_argument("--endpoint", default="/rag", help="API endpoint")
    parser.add_argument("--concurrency", type=int, default=50, help="Number of concurrent requests")
    parser.add_argument("--queries", default=None, help="File containing queries (one per line)")
    parser.add_argument("--num-queries", type=int, default=500, help="Number of queries to generate if no file provided")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between request batches (seconds)")
    parser.add_argument("--output", default="rag_test", help="Output file prefix")
    
    args = parser.parse_args()
    
    if args.queries:
        queries = load_queries_from_file(args.queries)
        print(f"Loaded {len(queries)} queries from {args.queries}")
    else:
        queries = generate_sample_queries(1000)
        print(f"Generated {len(queries)} sample queries")
    
    await run_real_test(args, queries)

if __name__ == "__main__":
    asyncio.run(main())
