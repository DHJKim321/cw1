import numpy as np
import json
import cupy as cp


import time

from task import our_knn_np, our_knn, our_knn_nearest_batch

def read_data(file_path=""):
    """
    Read data from a file
    """
    if file_path == "":
        return None
    if file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def testdata_kmeans(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        K = 10
        return N, D, A, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            K = data["k"]
            A = np.loadtxt(A_file)
        return N, D, A, K

def testdata_knn(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            A = np.loadtxt(A_file)
            X = np.loadtxt(X_file)
        return N, D, A, X, K
    
def testdata_ann(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            A = np.loadtxt(A_file)
            X = np.loadtxt(X_file)
        return N, D, A, X, K
    



def test_knn_performance(test_file=""):
    """
    Tests Top-K Nearest Neighbors performance on CPU (NumPy) and GPU (CuPy).

    Args:
        test_file (str): Path to JSON test file or "" for random data.

    Returns:
        None (Prints execution times)
    """
    # Load test data
    N, D, A, X, K = testdata_knn(test_file)

    print(f"\n=== Testing Top-K Nearest Neighbors for N={N}, D={D}, K={K} ===")

    # Convert A and X to NumPy (CPU) and CuPy (GPU) formats
    A_cpu, X_cpu = A.astype(np.float32), X.astype(np.float32)
    A_gpu, X_gpu = cp.asarray(A_cpu), cp.asarray(X_cpu)

    # ---- Test CPU (NumPy) Execution ----
    start_cpu = time.time()
    nearest_indices_cpu = our_knn_np(N, D, A_cpu, X_cpu, K, "l2")
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"CPU (NumPy) Execution Time: {cpu_time:.4f} sec")

    # ---- Test GPU (CuPy) Execution ----
    if N <= 100000:  # Direct execution for small data
        start_gpu = time.time()
        nearest_indices_gpu = our_knn(N, D, A_gpu, X_gpu, K, "l2", use_kernel=True)
        end_gpu = time.time()
        gpu_time = end_gpu - start_gpu
        print(f"⚡ GPU (CuPy) Execution Time: {gpu_time:.4f} sec")
    
    # ---- Test GPU Batch Execution for Large Data ----
    else:
        start_gpu = time.time()
        nearest_indices_gpu = our_knn_nearest_batch(N, D, A_gpu, X_gpu, K, batch_size=100000, distance_metric="l2")
        end_gpu = time.time()
        gpu_time = end_gpu - start_gpu
        print(f"🚀 GPU (Batch Mode) Execution Time: {gpu_time:.4f} sec")

    # ---- Speedup Comparison ----
    speedup = cpu_time / gpu_time
    print(f"💡 Speedup: {speedup:.2f}x (GPU vs. CPU)")

    # ---- Validate Results (Check if GPU & CPU Indices Match) ----
    assert np.allclose(cp.asnumpy(nearest_indices_gpu), nearest_indices_cpu), "Mismatch in Top-K indices!"
    print("Results Match: GPU & CPU return the same nearest neighbors!")

