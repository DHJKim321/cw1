import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
import scipy
from test import testdata_kmeans, testdata_knn, testdata_ann

# You can create any kernel here
subtract_square = cp.ElementwiseKernel('float32 x, float32 y', 
                                    'float32 z', 
                                    '''
                                    z = (x - y);
                                       z = z * z;                                   
                                    '''
                                    )

subtract_abs = cp.ElementwiseKernel('float32 x, float32 y',
                                    'float32 z',
                                    '''
                                    z = abs(x - y)
                                    ''')

sum_sqrt = cp.ReductionKernel('float32 x', 'float32 y', 'x', 'a + b', 'y = sqrt(a)', '0')

multiply = cp.ElementwiseKernel('float32 x, float32 y', 
                                'float32 z', 
                                '''z = x * y''')

_sum = cp.ReductionKernel('float32 x', 'float32 y', 'x', 'a + b', 'y = a', '0')

square = cp.ElementwiseKernel('float32 x', 'float32 y', '''y = x * x''')

divide = cp.ElementwiseKernel('float32 x, float32 y', 'float32 z', '''z = x / y''')

# def distance_cosine(X, Y, use_kernel=True):
#     if use_kernel:
#         sum_X = sum_sqrt(square(X))
#         sum_Y = sum_sqrt(square(Y))
#         dot = _sum(multiply(X, Y))
#         Z = multiply(sum_X, sum_Y)
#         W = divide(dot, Z)
#         V = 1 - W
#     else:
#         sum_X = cp.linalg.norm(X)
#         sum_Y = cp.linalg.norm(Y)
#         dot = cp.dot(X, Y)
#         W = cp.divide(dot, (sum_X * sum_Y))
#         V = 1 - W
#     return V

def distance_cosine(A, X, use_kernel=True):
    if use_kernel:
        sum_A = sum_sqrt(square(A), axis=1)  # Norm of A (N,)
        sum_X = sum_sqrt(square(X), axis=1)  # Norm of X (1,)
        dot = _sum(multiply(A, X), axis=1)  # Dot product (N,)
        Z = multiply(sum_A, sum_X)  # Multiply norms
        W = divide(dot, Z)  # Normalize dot product
        V = 1 - W  # Cosine distance (N,)
    else:
        sum_A = cp.linalg.norm(A, axis=1)
        sum_X = cp.linalg.norm(X, axis=1)
        dot = cp.sum(A * X, axis=1)
        W = dot / (sum_A * sum_X)
        V = 1 - W
    return V


def distance_cosine_streams(X, Y, use_kernel=True):
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()
    if use_kernel:
        with stream1:
            sum_X = sum_sqrt(square(X))
        with stream2:
            sum_Y = sum_sqrt(square(Y))
        Z = multiply(sum_X, sum_Y)
        with stream3:
            dot = _sum(multiply(X, Y))
        W = divide(dot, Z)
        V = 1 - W
    else:
        with stream1:
            sum_X = cp.linalg.norm(X)
        with stream2:
            sum_Y = cp.linalg.norm(Y)
        with stream3:
            dot = cp.dot(X, Y)
        W = cp.divide(dot, (sum_X * sum_Y))
        V = 1 - W
    return V


# def distance_l2(X, Y, use_kernel=True):
#     if use_kernel:
#         W = subtract_square(X, Y)
#         V = sum_sqrt(W) 
#     else:
#         V = cp.linalg.norm(X - Y) 
#     return V

# def distance_dot(X, Y, use_kernel=True):
#     if use_kernel:
#         Z = multiply(X, Y)
#         W = _sum(Z)
#     else:
#         W = cp.dot(X, Y)
#     return W

# def distance_manhattan(X, Y, use_kernel=True):
#     if use_kernel:
#         Z = subtract_abs(X, Y)
#         U = _sum(Z)
#     else:
#         U = cp.sum(cp.abs(X - Y))
#     return U

def distance_l2(A, X, use_kernel=True):
    if use_kernel:
        W = subtract_square(A, X)  # Element-wise squared difference (N, D)
        V = sum_sqrt(W, axis=1)  # Sum across D and take sqrt (N,)
    else:
        V = cp.linalg.norm(A - X, axis=1)  # GPU-accelerated L2 norm
    return V

def distance_manhattan(A, X, use_kernel=True):
    if use_kernel:
        Z = subtract_abs(A, X)  # Element-wise absolute difference (N, D)
        U = _sum(Z, axis=1)  # Sum across D (N,)
    else:
        U = cp.sum(cp.abs(A - X), axis=1)  # Use CuPy's optimized sum
    return U

def distance_dot(A, X, use_kernel=True):
    if use_kernel:
        Z = multiply(A, X)  # Element-wise multiplication (N, D)
        W = _sum(Z, axis=1)  # Sum across D (N,)
    else:
        W = cp.sum(A * X, axis=1)  # Efficiently compute dot product
    return W


def distance_cosine_np(X, Y):
    sum_X = np.linalg.norm(X)
    sum_Y = np.linalg.norm(Y)
    dot = np.dot(X, Y)
    W = np.divide(dot, (sum_X * sum_Y))
    V = 1 - W
    return V

def distance_l2_np(X, Y):
    return np.linalg.norm(X - Y) 

def distance_dot_np(X, Y):
    return np.dot(X, Y)

def distance_manhattan_np(X, Y):
    return np.sum(np.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K, distance_metric="l2", use_kernel = True):
    """_knn

    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (list[list[float]]): A collection of vectors(N x D)
        X (list[float]): A specified vector(ie. query vector)
        K (int): topK nearest neighbors to find
        distance_metric (str, optional): _description_. Defaults to "l2".
        use_kernel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Shape mismatch: A should be (N, D) and X should be (D,)")

    # Compute distances based on chosen metric
        
        # if distance_metric == "cosine":
        #     distances = distance_cosine(A, X, use_kernel)
        # elif distance_metric == "l2":
        #     distances = distance_l2(A, X, use_kernel)
        # elif distance_metric == "dot":
        #     distances = -distance_dot(A, X, use_kernel)
        # elif distance_metric == "manhattan":
        #     distances = distance_manhattan(A, X, use_kernel) 
    if distance_metric == "cosine":
        distances = distance_cosine(A, X[None, :], use_kernel)  # Broadcast X across all rows of A
    elif distance_metric == "l2":
        distances = distance_l2(A, X[None, :], use_kernel)  # Apply L2 distance using kernel
    elif distance_metric == "dot":
        distances = -distance_dot(A, X[None, :], use_kernel)  # Apply dot product distance
    elif distance_metric == "manhattan":
        distances = distance_manhattan(A, X[None, :], use_kernel)  # Apply Manhattan distance
    else:
        raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

        # Get the indices of the top K smallest distances
    top_k_indices = cp.argsort(distances)[:K]

    return top_k_indices

    
def our_knn_np(N, D, A, X, K, distance_metric="l2"):
  
    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Shape mismatch: A should be (N, D) and X should be (D,)")

    # Compute distances based on chosen metric
    if distance_metric == "cosine":
        distances =  np.array([distance_cosine_np(A[i], X) for i in range(N)])
    elif distance_metric == "l2":
        distances = np.array([distance_l2_np(A[i], X) for i in range(N)])
    elif distance_metric == "dot":
        distances = -np.array([distance_dot_np(A[i], X) for i in range(N)])
    elif distance_metric == "manhattan":
        distances = np.array([distance_manhattan_np(A[i], X) for i in range(N)])
    else:
        raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

    # Get the indices of the top K smallest distances
    top_k_indices = np.argsort(distances)[:K]

    return top_k_indices




def our_knn_nearest_batch(N, D, A, X, K, batch_size=100000, distance_metric="l2", use_kernel=True):

    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Shape mismatch: A should be (N, D) and X should be (D,)")

    top_k_results = []
    top_k_distances = []

    for i in range(0, N, batch_size):
        batch_A = A[i:i+batch_size]  # Extract batch

        if distance_metric == "cosine":
            distances = distance_cosine(batch_A, X[None, :], use_kernel)  # Broadcast X across all rows of A
        elif distance_metric == "l2":
            distances = distance_l2(batch_A, X[None, :], use_kernel)  # Apply L2 distance using kernel
        elif distance_metric == "dot":
            distances = -distance_dot(batch_A, X[None, :], use_kernel)  # Apply dot product distance
        elif distance_metric == "manhattan":
            distances = distance_manhattan(batch_A, X[None, :], use_kernel)  # Apply Manhattan distance
        else:
            raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

        # Get Top-K from this batch
        batch_top_k = cp.argsort(distances)[:K]
        batch_top_k_distances = distances[batch_top_k]

        # Adjust indices for batch offset
        top_k_results.append(batch_top_k + i)
        top_k_distances.append(batch_top_k_distances)

    # Merge Top-K results from all batches
    top_k_results = cp.concatenate(top_k_results)
    top_k_distances = cp.concatenate(top_k_distances)

    # Get final Top-K across all batches
    final_top_k = cp.argsort(top_k_distances)[:K]

    return top_k_results[final_top_k] 




# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

mean_kernel = cp.ReductionKernel(
    'float32 x',  # Input type
    'float32 y',  # Output type
    'x',          # Map function (identity function here)
    'a + b',      # Reduce function (sum)
    'y = a / _ind.size()',  # Post-reduction function (mean)
    '0',          # Identity value for the reduction (sum)
    'mean_kernel' # Kernel name
)
def our_kmeans(N, D, A, K, use_kernel=True):
    """
    Input:
    N (int): Number of vectors.
    D (int): Dimension of each vector.
    A (list[list[float]]): Collection of vectors (N x D).
    K (int): Number of clusters.

    Returns:
    list[int]: Cluster IDs for each vector.
    list[list[float]]: Centroids of each cluster.
    """
    
    A = cp.asarray(A)
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices, :] 
    
    for _ in range(100):
        distance_list = []
        for centroid in centroids:
            diff_sq = subtract_square(A, centroid)  # Why not use diff_sq = _sum(substract_square(A, centroid)) using kernel?
            d = cp.sqrt(cp.sum(diff_sq, axis=1))     
            distance_list.append(d[:, cp.newaxis])
        
        distances = cp.concatenate(distance_list, axis=1)  
        labels = cp.argmin(distances, axis=1)
        
        if use_kernel:
            new_centroids = cp.array([mean_kernel(A[labels == k], axis=0) if cp.any(labels == k) 
                                else centroids[k] for k in range(K)])
        else:
            new_centroids = cp.array([cp.mean(A[labels == k], axis=0) if cp.any(labels == k) 
                                else centroids[k] for k in range(K)])
        
        if cp.allclose(centroids, new_centroids):
            print("Centroids have converged. Stopping iterations.")
            break
        
        centroids = new_centroids

    return labels, centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K, use_kernel=True):
    """_ann

    Args:
        N (int): Number of vectors 
        D (int): Dimension of vectors
        A (list[list[float]]): A collection of vectors(N x D)
        X (list[float]): A specified vector(ie. query vector)
        K (int): Top K nearest neighbors to find
        
    Returns:
        Result[K]: Top K nearest neighbors ID (index of the vector in A)
    """
    n_probe = 3 # the number of clusters
    cluster_ids, centroids = our_kmeans(N, D, A, n_probe, use_kernel=use_kernel)

    top_k_indices = []
        
    label = cp.argmin(distance_l2(centroids, X, use_kernel=use_kernel))
    cluster = A[cluster_ids == label]
    top_k_indices = our_knn(cluster.shape[0], D, cluster, X, K, use_kernel=use_kernel)
    
    return top_k_indices
        
    

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_cosine(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_cosine(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold], rtol=1e-06, atol=1e-6)
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_cosine_streams(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_cosine_streams(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold], rtol=1e-06, atol=1e-6)
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_l2(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_l2(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = cp.linalg.norm(X - Y)
    assert cp.isclose([ours], [gold])
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_dot(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_dot(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = cp.dot(X, Y)
    assert cp.isclose([ours], [gold])
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_manhattan(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_manhattan(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = scipy.spatial.distance.cityblock(X.get(), Y.get())
    assert cp.isclose([ours], [gold])
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_cosine_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_cosine(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_cosine_np(X, Y)
    end_cpu = time.time()
    # print(f"Cosine (GPU): {end_gpu-start_gpu}")
    # print(f"Cosine (CPU): {end_cpu-start_cpu}")
    # print(f"Cosine (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

def test_l2_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_l2(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_l2_np(X, Y)
    end_cpu = time.time()
    # print(f"L2 (GPU): {end_gpu-start_gpu}")
    # print(f"L2 (CPU): {end_cpu-start_cpu}")
    # print(f"L2 (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

def test_dot_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_dot(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_dot_np(X, Y)
    end_cpu = time.time()
    # print(f"Dot (GPU): {end_gpu-start_gpu}")
    # print(f"Dot (CPU): {end_cpu-start_cpu}")
    # print(f"Dot (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

def test_manhattan_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_manhattan(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_manhattan_np(X, Y)
    end_cpu = time.time()
    # print(f"Manhattan (GPU): {end_gpu-start_gpu}")
    # print(f"Manhattan (CPU): {end_cpu-start_cpu}")
    # print(f"Manhattan (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_our_ann():
    # Generate test data
    N, D, A, X, K = testdata_ann()
    
    # Convert data to CuPy arrays
    A_cp = cp.asarray(A)
    X_cp = cp.asarray(X)
    
    # Run the our_ann function
    top_k_indices = our_ann(N, D, A_cp, X_cp, K)
    
    # Convert the result back to NumPy for assertion
    top_k_indices_np = cp.asnumpy(top_k_indices)
    
    # Check the length of the result
    assert len(top_k_indices_np) == K, f"Expected {K} indices, but got {len(top_k_indices_np)}"
    
    # Check if the indices are within the valid range
    assert np.all(top_k_indices_np < N), "Some indices are out of range"
    
    print("top_k_indices_np.shape:", top_k_indices_np.shape)
    print("top_k_indices_np:", top_k_indices_np)
    
    print("Test passed!")
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)


def test_knn_performance(test_file="", dis_metric="l2"):
    """
    Tests Top-K Nearest Neighbors performance on CPU (NumPy) and GPU (CuPy).

    distance metrics = "l2", "cosine", "dot" or "manhattan"

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
    nearest_indices_cpu = our_knn_np(N, D, A_cpu, X_cpu, K, dis_metric)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"CPU (NumPy) Execution Time: {cpu_time:.4f} sec")

    # ---- Test GPU (CuPy) Execution ----
    if N <= 100000:  # Direct execution for small data
        start_gpu = time.time()
        nearest_indices_gpu = our_knn(N, D, A_gpu, X_gpu, K, dis_metric, use_kernel=True)
        end_gpu = time.time()
        gpu_time = end_gpu - start_gpu
        print(f"⚡ GPU (CuPy) Execution Time: {gpu_time:.4f} sec")
    
    # ---- Test GPU Batch Execution for Large Data ----
    else:
        start_gpu = time.time()
        nearest_indices_gpu = our_knn_nearest_batch(N, D, A_gpu, X_gpu, K, batch_size=100000, distance_metric=dis_metric, use_kernel=True)
        end_gpu = time.time()
        gpu_time = end_gpu - start_gpu
        print(f"GPU (Batch Mode) Execution Time: {gpu_time:.4f} sec")

    # ---- Speedup Comparison ----
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x (GPU vs. CPU)")

    # ---- Validate Results (Check if GPU & CPU Indices Match) ----
    assert np.allclose(cp.asnumpy(nearest_indices_gpu), nearest_indices_cpu), "Mismatch in Top-K indices!"
    print("Results Match: GPU & CPU return the same nearest neighbors!")




if __name__ == "__main__":
    ### Test Distance Functions
    # dimensions = [2, 2**15]
    # N = 30
    # print(f"N={N}")
    # print("Testing Distance Functions")
    # for D in dimensions:
    #     print(f"Dimension: {D}")
    #     cosine_kernel_list, cosine_api_list = cp.empty(N), cp.empty(N)
    #     cosine_kernel_stream_list, cosine_api_stream_list = cp.empty(N), cp.empty(N)
    #     l2_kernel_list, l2_api_list = cp.empty(N), cp.empty(N)
    #     dot_kernel_list, dot_api_list = cp.empty(N), cp.empty(N)
    #     manhattan_kernel_list, manhattan_api_list = cp.empty(N), cp.empty(N)
    #     for i in range(N):
    #         cosine_kernel = test_cosine(D)
    #         cosine_kernel_list[i] = cosine_kernel
    #         cosine_api = test_cosine(D, use_kernel=False)
    #         cosine_api_list[i] = cosine_api
    #         cosine_kernel_stream = test_cosine_streams(D, use_kernel=True)
    #         cosine_kernel_stream_list[i] = cosine_kernel_stream
    #         cosine_api_stream = test_cosine_streams(D, use_kernel=False)
    #         cosine_api_stream_list[i] = cosine_api_stream
    #         l2_kernel = test_l2(D)
    #         l2_kernel_list[i] = l2_kernel
    #         l2_api = test_l2(D, use_kernel=False)
    #         l2_api_list[i] = l2_api
    #         dot_kernel = test_dot(D)
    #         dot_kernel_list[i] = dot_kernel
    #         dot_api = test_dot(D, use_kernel=False)
    #         dot_api_list[i] = dot_api
    #         manhattan_kernel = test_manhattan(D)
    #         manhattan_kernel_list[i] = manhattan_kernel
    #         manhattan_api = test_manhattan(D, use_kernel=False)
    #         manhattan_api_list[i] = manhattan_api
    #     print("----------------------------------------")
    #     print("Absolute Runtime Values (API)")
    #     print(f"Cosine (Stream): {cosine_api_stream_list.mean()}")
    #     print(f"Cosine (Without Stream): {cosine_api_list.mean()}")
    #     print(f"L2: {l2_api_list.mean()}")
    #     print(f"Dot: {dot_api_list.mean()}")
    #     print(f"Manhattan: {manhattan_api_list.mean()}")
    #     print("----------------------------------------")
    #     print("Absolute Runtime Values (Kernel)")
    #     print(f"Cosine (Stream): {cosine_kernel_stream_list.mean()}")
    #     print(f"Cosine (Without Stream): {cosine_kernel_list.mean()}")
    #     print(f"L2: {l2_kernel_list.mean()}")
    #     print(f"Dot: {dot_kernel_list.mean()}")
    #     print(f"Manhattan: {manhattan_kernel_list.mean()}")
    #     print("----------------------------------------")
    #     print("Differences in Speed (Positive means API is faster than Kernel)")
    #     print(f"Cosine Difference: {cosine_kernel_list.mean()-cosine_api_list.mean()}")
    #     print(f"Cosine Difference (Streams): {cosine_kernel_stream_list.mean()-cosine_api_stream_list.mean()}")
    #     print(f"L2 Difference: {l2_kernel_list.mean()-l2_api_list.mean()}")
    #     print(f"Dot Difference: {dot_kernel_list.mean()-dot_api_list.mean()}")
    #     print(f"Manhattan Difference: {manhattan_kernel_list.mean()-manhattan_api_list.mean()}")
    #     print("----------------------------------------")
    # print(f"Testing Differences Between CPU and GPU")
    # for D in dimensions:
    #     print(f"Dimension: {D}")
    #     cosine_gpu, cosine_cpu = cp.empty(N), cp.empty(N)
    #     l2_gpu, l2_cpu = cp.empty(N), cp.empty(N)
    #     dot_gpu, dot_cpu = cp.empty(N), cp.empty(N)
    #     manhattan_gpu, manhattan_cpu = cp.empty(N), cp.empty(N)
    #     for i in range(N):
    #         gpu, cpu = test_cosine_gpu_vs_cpu(D) # diff = cpu - gpu
    #         cosine_gpu[i] = gpu
    #         cosine_cpu[i] = cpu
    #         gpu, cpu = test_l2_gpu_vs_cpu(D)
    #         l2_gpu[i] = gpu
    #         l2_cpu[i] = cpu
    #         gpu, cpu = test_dot_gpu_vs_cpu(D)
    #         dot_gpu[i] = gpu
    #         dot_cpu[i] = cpu
    #         gpu, cpu = test_manhattan_gpu_vs_cpu(D)
    #         manhattan_gpu[i] = gpu
    #         manhattan_cpu[i] = cpu
    #     print(f"Cosine CPU: {cosine_cpu.mean()}")
    #     print(f"Cosine GPU: {cosine_gpu.mean()}")
    #     print(f"Cosine CPU - GPU: {(cosine_cpu.mean() - cosine_gpu.mean()).item()}")
    #     print(f"L2 CPU: {l2_cpu.mean()}")
    #     print(f"L2 GPU: {l2_gpu.mean()}")
    #     print(f"L2 CPU - GPU: {(l2_cpu.mean() - l2_gpu.mean()).item()}")
    #     print(f"Dot CPU: {dot_cpu.mean()}")
    #     print(f"Dot GPU: {dot_gpu.mean()}")
    #     print(f"Dot CPU - GPU: {(dot_cpu.mean() - dot_gpu.mean()).item()}")
    #     print(f"Manhattan CPU: {cosine_cpu.mean()}")
    #     print(f"Manhattan GPU: {manhattan_gpu.mean()}")
    #     print(f"Manhattan CPU - GPU: {(manhattan_cpu.mean() - manhattan_gpu.mean()).item()}")
    #     print("----------------------------------------")


    # X = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    # Y = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)

    # print("Cosine Distance:", distance_cosine(X, Y, use_kernel=True))
    # print("L2 Distance:", distance_l2(X, Y, use_kernel=True))
    # print("Dot Product Distance:", distance_dot(X, Y, use_kernel=True))
    # print("Manhattan Distance:", distance_manhattan(X, Y, use_kernel=True))

    # N, D, K = 100, 2, 10  # 10 vectors, each of dimension 3, find top 3 nearest neighbors

    # # Generate random test data
    # cp.random.seed(42)
    # # A_np = np.random.randn(N, D)
    # # X_np = np.random.randn(D)
    # A = cp.random.randn(N, D).astype(cp.float32)  # Random dataset of 10 vectors
    # X = cp.random.randn(D).astype(cp.float32)  # Random query vector

    # # Test our_knn function for different distance metrics
    # for metric in ["l2", "cosine", "dot", "manhattan"]:
    #     top_k_indices = our_knn(N, D, A, X, K, distance_metric=metric, use_kernel=True)
    #     print(f"Top {K} nearest neighbors using {metric} distance:", cp.asnumpy(top_k_indices))
    #     top_k_indices_cp = our_knn(N, D, A, X, K, distance_metric=metric, use_kernel=False)
    #     print(f"Top {K} nearest neighbors using {metric} distance for cp:", cp.asnumpy(top_k_indices_cp))
    #     # top_k_indices_np = our_knn_np(N, D, A_np, X_np, K, distance_metric=metric)
    #     # print(f"Top {K} nearest neighbors using {metric} distance for np:", cp.asnumpy(top_k_indices_np))

    # Set parameters
    N, D, K = 4000, 2, 10

    # Generate random dataset
    A_gpu = cp.random.randn(N, D).astype(cp.float32)  # GPU array
    X_gpu = cp.random.randn(D).astype(cp.float32)

    # Convert to NumPy for CPU testing
    A_cpu = cp.asnumpy(A_gpu)
    X_cpu = cp.asnumpy(X_gpu)

    # Test different distance metrics
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        # 1️⃣ Measure CPU Time (NumPy)
        start_cpu = time.time()
        top_k_indices_cpu = our_knn_np(N, D, A_cpu, X_cpu, K, distance_metric=metric)
        end_cpu = time.time()
        cpu_time = end_cpu - start_cpu

        # 2️⃣ Measure GPU Time (CuPy - Without Kernel Optimization)
        start_cp = time.time()
        top_k_indices_cp = our_knn(N, D, A_gpu, X_gpu, K, distance_metric=metric, use_kernel=False)
        end_cp = time.time()
        cp_time = end_cp - start_cp

        # 3️⃣ Measure GPU Time (CuPy - With Kernel Optimization)
        start_elm = time.time()
        top_k_indices_element = our_knn(N, D, A_gpu, X_gpu, K, distance_metric=metric, use_kernel=True)
        end_elm = time.time()
        elm_time = end_elm - start_elm

        # Print results
        print(f"⚡ {metric.upper()} Distance Results:")
        print(f"    CPU Time (NumPy): {cpu_time:.6f} sec")
        print(f"    GPU Time (CuPy - No Kernel): {cp_time:.6f} sec")
        print(f"    GPU Time (CuPy - Kernel): {elm_time:.6f} sec")
        print(f"    Speedup (CPU vs CuPy No Kernel): {round(cpu_time / cp_time, 2)}x")
        print(f"    Speedup (CPU vs CuPy Kernel): {round(cpu_time / elm_time, 2)}x")
        print(f"    Speedup (CuPy No Kernel vs Kernel): {round(cp_time / elm_time, 2)}x\n")



    ### Test KNN
    ### Test KMeans
    
    ### Test Ann
    # test_our_ann()