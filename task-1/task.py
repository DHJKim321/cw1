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

cosine_kernel = cp.RawKernel(r'''
extern "C" __global__
void cosine_similarity(float* A, float* B, float* result, int N) {
    __shared__ float shared_A[256];  
    __shared__ float shared_B[256];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;  // Prevent out-of-bounds access

    shared_A[threadIdx.x] = A[tid];
    shared_B[threadIdx.x] = B[tid];
    __syncthreads();

    float dot = 0.0, normA = 0.0, normB = 0.0;
    dot += shared_A[threadIdx.x] * shared_B[threadIdx.x];
    normA += shared_A[threadIdx.x] * shared_A[threadIdx.x];
    normB += shared_B[threadIdx.x] * shared_B[threadIdx.x];

    __syncthreads();

    if (tid < N) {
        result[tid] = 1.0 - (dot / (sqrt(normA) * sqrt(normB) + 1e-8));
    }
}
''', 'cosine_similarity')

def cosine_similarity_gpu(X, Y):
    N = X.shape[0]
    num_blocks = max(1, int((N + 255) // 256))
    result = cp.zeros(N, dtype=cp.float32)
    cosine_kernel((num_blocks,), (256,), (X, Y, result, N))
    return result

l2_kernel = cp.RawKernel(r'''
extern "C" __global__
void l2_distance(float* A, float* B, float* result, int N) {
    __shared__ float shared_A[256];  
    __shared__ float shared_B[256];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;  

    shared_A[threadIdx.x] = A[tid];
    shared_B[threadIdx.x] = B[tid];
    __syncthreads();

    float diff = shared_A[threadIdx.x] - shared_B[threadIdx.x];
    result[tid] = sqrt(diff * diff);
}
''', 'l2_distance')

def l2_distance_gpu(X, Y):
    N = X.shape[0]
    num_blocks = max(1, int((N + 255) // 256))
    result = cp.zeros(N, dtype=cp.float32)
    l2_kernel((num_blocks,), (256,), (X, Y, result, N))
    return result

dot_kernel = cp.RawKernel(r'''
extern "C" __global__
void dot_product(float* A, float* B, float* result, int N) {
    __shared__ float shared_A[256];  
    __shared__ float shared_B[256];  
    __shared__ float partial_sum[256];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    shared_A[threadIdx.x] = A[tid];
    shared_B[threadIdx.x] = B[tid];
    __syncthreads();

    partial_sum[threadIdx.x] = shared_A[threadIdx.x] * shared_B[threadIdx.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, partial_sum[0]);
    }
}
''', 'dot_product')

def dot_product_gpu(X, Y):
    N = X.shape[0]
    num_blocks = max(1, int((N + 255) // 256))
    result = cp.zeros(1, dtype=cp.float32)
    dot_kernel((num_blocks,), (256,), (X, Y, result, N))
    return result

manhattan_kernel = cp.RawKernel(r'''
extern "C" __global__
void manhattan_distance(float* A, float* B, float* result, int N) {
    __shared__ float shared_A[256];  
    __shared__ float shared_B[256];  
    __shared__ float partial_sum[256];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    shared_A[threadIdx.x] = A[tid];
    shared_B[threadIdx.x] = B[tid];
    __syncthreads();

    partial_sum[threadIdx.x] = abs(shared_A[threadIdx.x] - shared_B[threadIdx.x]);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, partial_sum[0]);
    }
}
''', 'manhattan_distance')

def manhattan_distance_gpu(X, Y):
    N = X.shape[0]
    num_blocks = max(1, int((N + 255) // 256))
    result = cp.zeros(1, dtype=cp.float32)
    manhattan_kernel((num_blocks,), (256,), (X, Y, result, N))
    return result

def distance_cosine(X, Y, use_kernel=True):
    if use_kernel:
        sum_X = sum_sqrt(square(X))
        sum_Y = sum_sqrt(square(Y))
        dot = _sum(multiply(X, Y))
        Z = multiply(sum_X, sum_Y)
        W = divide(dot, Z)
        V = 1 - W
    else:
        sum_X = cp.linalg.norm(X)
        sum_Y = cp.linalg.norm(Y)
        dot = cp.dot(X, Y)
        W = cp.divide(dot, (sum_X * sum_Y))
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


def distance_l2(X, Y, use_kernel=True):
    if use_kernel:
        W = subtract_square(X, Y)
        V = sum_sqrt(W) 
    else:
        V = cp.linalg.norm(X - Y) 
    return V

def distance_dot(X, Y, use_kernel=True):
    if use_kernel:
        Z = multiply(X, Y)
        W = _sum(Z)
    else:
        W = cp.dot(X, Y)
    return W

def distance_manhattan(X, Y, use_kernel=True):
    if use_kernel:
        Z = subtract_abs(X, Y)
        U = _sum(Z)
    else:
        U = cp.sum(cp.abs(X - Y))
    return U

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
    if use_kernel:
        if distance_metric == "cosine":
            distances = distance_cosine(A, X, use_kernel)
        elif distance_metric == "l2":
            distances = distance_l2(A, X, use_kernel)
        elif distance_metric == "dot":
            distances = -distance_dot(A, X, use_kernel)
        elif distance_metric == "manhattan":
            distances = distance_manhattan(A, X, use_kernel) 
        else:
            raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")
    else:
        if distance_metric == "cosine":
            distances = cp.array([distance_cosine(A[i], X, use_kernel) for i in range(N)])
        elif distance_metric == "l2":
            distances = cp.array([distance_l2(A[i], X, use_kernel) for i in range(N)])
        elif distance_metric == "dot":
            distances = -cp.array([distance_dot(A[i], X, use_kernel) for i in range(N)])
        elif distance_metric == "manhattan":
            distances = cp.array([distance_manhattan(A[i], X, use_kernel) for i in range(N)])
        else:
            raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

        # Get the indices of the top K smallest distances
    top_k_indices = cp.argsort(distances)[:K]

    return top_k_indices

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

def test_cosine_raw(D=2):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = cosine_similarity_gpu(X, Y)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold], rtol=1e-06, atol=1e-6)
    return end-start

def test_l2_raw(D=2):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = l2_distance_gpu(X, Y)
    end = time.time()
    gold = cp.linalg.norm(X - Y)
    assert cp.isclose([ours], [gold])
    return end-start

def test_dot_raw(D=2):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = dot_product_gpu(X, Y)
    end = time.time()
    gold = cp.dot(X, Y)
    assert cp.isclose([ours], [gold])
    return end-start

def test_manhattan_raw(D=2):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time
    ours = manhattan_distance_gpu(X, Y)
    end = time.time()
    gold = scipy.spatial.distance.cityblock(X.get(), Y.get())
    assert cp.isclose([ours], [gold])
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

if __name__ == "__main__":
    ### Test Distance Functions
    dimensions = [2, 2**15]
    N = 30
    print(f"N={N}")
    print("Testing Distance Functions")
    for D in dimensions:
        print(f"Dimension: {D}")
        cosine_kernel_list, cosine_api_list, cosine_raw_list = cp.empty(N), cp.empty(N), cp.empty(N)
        cosine_kernel_stream_list, cosine_api_stream_list = cp.empty(N), cp.empty(N)
        l2_kernel_list, l2_api_list, l2_raw_list = cp.empty(N), cp.empty(N), cp.empty(N)
        dot_kernel_list, dot_api_list, dot_raw_list = cp.empty(N), cp.empty(N), cp.empty(N)
        manhattan_kernel_list, manhattan_api_list, manhattan_raw_list = cp.empty(N), cp.empty(N), cp.empty(N)
        for i in range(N):
            cosine_kernel = test_cosine(D)
            cosine_kernel_list[i] = cosine_kernel
            cosine_api = test_cosine(D, use_kernel=False)
            cosine_api_list[i] = cosine_api
            cosine_raw = test_cosine_raw(D)
            cosine_raw_list[i] = cosine_raw
            cosine_kernel_stream = test_cosine_streams(D, use_kernel=True)
            cosine_kernel_stream_list[i] = cosine_kernel_stream
            cosine_api_stream = test_cosine_streams(D, use_kernel=False)
            cosine_api_stream_list[i] = cosine_api_stream
            l2_kernel = test_l2(D)
            l2_kernel_list[i] = l2_kernel
            l2_api = test_l2(D, use_kernel=False)
            l2_api_list[i] = l2_api
            l2_raw = test_l2_raw(D)
            l2_raw_list[i] = l2_raw
            dot_kernel = test_dot(D)
            dot_kernel_list[i] = dot_kernel
            dot_api = test_dot(D, use_kernel=False)
            dot_api_list[i] = dot_api
            dot_raw = test_dot_raw(D)
            dot_raw_list[i] = dot_raw
            manhattan_kernel = test_manhattan(D)
            manhattan_kernel_list[i] = manhattan_kernel
            manhattan_api = test_manhattan(D, use_kernel=False)
            manhattan_api_list[i] = manhattan_api
            manhattan_raw = test_manhattan_raw(D)
            manhattan_raw_list[i] = manhattan_raw
        print("----------------------------------------")
        print("Absolute Runtime Values (API)")
        print(f"Cosine (Stream): {cosine_api_stream_list.mean()}")
        print(f"Cosine (Without Stream): {cosine_api_list.mean()}")
        print(f"L2: {l2_api_list.mean()}")
        print(f"Dot: {dot_api_list.mean()}")
        print(f"Manhattan: {manhattan_api_list.mean()}")
        print("----------------------------------------")
        print("Absolute Runtime Values (Kernel)")
        print(f"Cosine (Stream): {cosine_kernel_stream_list.mean()}")
        print(f"Cosine (Without Stream): {cosine_kernel_list.mean()}")
        print(f"L2: {l2_kernel_list.mean()}")
        print(f"Dot: {dot_kernel_list.mean()}")
        print(f"Manhattan: {manhattan_kernel_list.mean()}")
        print("----------------------------------------")
        print("Absolute Runtime Values (Raw)")
        print(f"Cosine: {cosine_raw_list.mean()}")
        print(f"L2: {l2_raw_list.mean()}")
        print(f"Dot: {dot_raw_list.mean()}")
        print(f"Manhattan: {manhattan_raw_list.mean()}")
        print("----------------------------------------")
        print("Differences in Speed (Positive means API is faster than Kernel)")
        print(f"Cosine Difference: {cosine_kernel_list.mean()-cosine_api_list.mean()}")
        print(f"Cosine Difference (Streams): {cosine_kernel_stream_list.mean()-cosine_api_stream_list.mean()}")
        print(f"L2 Difference: {l2_kernel_list.mean()-l2_api_list.mean()}")
        print(f"Dot Difference: {dot_kernel_list.mean()-dot_api_list.mean()}")
        print(f"Manhattan Difference: {manhattan_kernel_list.mean()-manhattan_api_list.mean()}")
        print("----------------------------------------")
    print(f"Testing Differences Between CPU and GPU")
    for D in dimensions:
        print(f"Dimension: {D}")
        cosine_gpu, cosine_cpu = cp.empty(N), cp.empty(N)
        l2_gpu, l2_cpu = cp.empty(N), cp.empty(N)
        dot_gpu, dot_cpu = cp.empty(N), cp.empty(N)
        manhattan_gpu, manhattan_cpu = cp.empty(N), cp.empty(N)
        for i in range(N):
            gpu, cpu = test_cosine_gpu_vs_cpu(D) # diff = cpu - gpu
            cosine_gpu[i] = gpu
            cosine_cpu[i] = cpu
            gpu, cpu = test_l2_gpu_vs_cpu(D)
            l2_gpu[i] = gpu
            l2_cpu[i] = cpu
            gpu, cpu = test_dot_gpu_vs_cpu(D)
            dot_gpu[i] = gpu
            dot_cpu[i] = cpu
            gpu, cpu = test_manhattan_gpu_vs_cpu(D)
            manhattan_gpu[i] = gpu
            manhattan_cpu[i] = cpu
        print(f"Cosine CPU: {cosine_cpu.mean()}")
        print(f"Cosine GPU: {cosine_gpu.mean()}")
        print(f"Cosine CPU - GPU: {(cosine_cpu.mean() - cosine_gpu.mean()).item()}")
        print(f"L2 CPU: {l2_cpu.mean()}")
        print(f"L2 GPU: {l2_gpu.mean()}")
        print(f"L2 CPU - GPU: {(l2_cpu.mean() - l2_gpu.mean()).item()}")
        print(f"Dot CPU: {dot_cpu.mean()}")
        print(f"Dot GPU: {dot_gpu.mean()}")
        print(f"Dot CPU - GPU: {(dot_cpu.mean() - dot_gpu.mean()).item()}")
        print(f"Manhattan CPU: {cosine_cpu.mean()}")
        print(f"Manhattan GPU: {manhattan_gpu.mean()}")
        print(f"Manhattan CPU - GPU: {(manhattan_cpu.mean() - manhattan_gpu.mean()).item()}")
        print("----------------------------------------")
    
    ### Test KNN
    ### Test KMeans
    
    ### Test Ann
    # test_our_ann()