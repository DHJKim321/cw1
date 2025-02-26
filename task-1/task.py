import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
import scipy
from scipy.cluster.vq import vq
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
    iter = 100 # the number of iterations of the k-means algorithm running.
    A = cp.asarray(A)
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices, :] 
    
    for _ in range(iter):
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

    return centroids, labels

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
    
    n_probe = 3 # the number of clusters searched during a query
    

    centroids, labels, = our_kmeans(N, D, A, K, use_kernel=use_kernel)

    top_k_indices = []
    for _ in range(n_probe):    
        label = cp.argmin(distance_l2(centroids, X, use_kernel=use_kernel))
        cluster = A[labels == label]
        top_k_indices = our_knn(cluster.shape[0], D, cluster, X, K, use_kernel=use_kernel)
    
    return top_k_indices
        
def our_ann_IVFPQ(N, D, A, X, K, M, n_probe, use_kernel=True):
    """Approximate Nearest Neighbor search using IVFPQ(Inverted Vectors for Product Quantization)
        Usually, M is 8
        The larger n_probe is, the more accurate the search is, but the slower the search is.
        Currently, distance metric is set to L2 distance, but you can use cosine distance or dot product as well.

    Args:
        N (int): Number of vectors 
        D (int): Dimension of vectors
        A (list[list[float]]): A collection of vectors(N x D)
        X (list[float]): A specified vector(ie. query vector)
        K (int): Top K nearest neighbors to find
        M (int): Number of sub-vectors of PQ subvectors
        n_probe (int): the number of clusters searched during a query
        
    Returns:
        Result[K]: Top K nearest neighbors ID (index of the vector in A)
    """
    dsub = D // M  # Dimensionality of sub-vector
    ksub = 256  # Number of Clusters for sub-vectors
    # Step 1: First-Level Clustering (Coarse Quantization - IVF)
    centroids, labels = our_kmeans(N, D, A, K, use_kernel=use_kernel)
    
    # Compute residual vectores (y - q1(y))
    residuals = A - centroids[labels]
    
    # Step 2: Product Quantization(Fine Quantization) Training
    pq_codebooks = np.empty((M, ksub, dsub), dtype=np.float32) # PQ centroids
    pq_codes = np.empty((N, M), dtype=np.uint8) # Encoded PQ codes
    
    for m in range(M):
        sub_vecs = residuals[:, m * dsub:(m + 1) * dsub] # Extract sub-vectors
        pq_codebooks[m], _ = our_kmeans(N, dsub, sub_vecs, ksub, use_kernel=use_kernel)
        pq_codes[:, m] = vq(sub_vecs, pq_codebooks[m]) # Assign to PQ centroids
        
    # Step 3: Query Processintg
    # Find the closest 'n_probe' clusters to the query vector
    closest_clusters = cp.argsort(distance_l2(centroids, X, use_kernel=use_kernel))[:n_probe]
    
    # Retrieve points in the selected clusters
    candidates = []
    for label in closest_clusters:
        candidates.extend(np.where(labels == label)[0])
        
    # If no candidates found, return an empty list
    if len(candidates) == 0:
        return []
    
    query_coarse_id = closest_clusters[0] # Assign query to closest cluster
    query_residual = X - centroids[query_coarse_id] # Compute residual vector for query
    
    # Step 4 : Compute PQ Distance Table
    query_pq = np.empty((M,), dtype=np.uint8)
    dist_table = np.empty((M, ksub), dtype=np.float32)
    
    for m in range(M):
        query_sub = query_residual[m * dsub:(m + 1) * dsub] # Extract query sub-vector
        dist_table[m, :] = distance_l2(pq_codebooks[m], query_sub, use_kernel=use_kernel)
        query_pq[m] = np.argmin(dist_table[m]) # Assign PQ code for query
        
    # Step 5: Approximate Nearest Neighbor Search Using our KNN
    distances = []
    for idx in candidates:
        pq_code = pq_codes[idx]

        # Compute final IVFPQ distance
        # d = || x - y_C - y_R ||²
        coarse_distance = np.sum((X - centroids[labels[idx]])**2)  # || x - q1(y) ||². q1(y) = y_C
        refined_distance = np.sum(dist_table[np.arange(M), pq_code])    # Lookup table sum for q2(y - q1(y)), q2(y - q1(y)) = y_R.
                                                                        # refined_distance is equivalent to || y_R ||² because dist_table is already squared.

        total_distance = coarse_distance + refined_distance
        distances.append((idx, total_distance))

    # Sort by distance and return top-K indices
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances[:K]]




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
    
def test_our_ann_IVFPQ():
    # Generate test data
    N, D, A, X, K, M, n_probe = testdata_ann()
    
    # Convert data to CuPy arrays
    A_cp = cp.asarray(A)
    X_cp = cp.asarray(X)
    
    # Run the our_ann_IVFPQ function
    top_k_indices = our_ann_IVFPQ(N, D, A_cp, X_cp, K, M, n_probe)
    
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
    # ### Test Distance Functions
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
    #     print(f"Cosine (Stream): {cp.median(cosine_api_stream_list)}")
    #     print(f"Cosine (Without Stream): {cp.median(cosine_api_list)}")
    #     print(f"L2: {cp.median(l2_api_list)}")
    #     print(f"Dot: {cp.median(dot_api_list)}")
    #     print(f"Manhattan: {cp.median(manhattan_api_list)}")
    #     print("----------------------------------------")
    #     print("Absolute Runtime Values (Kernel)")
    #     print(f"Cosine (Stream): {cp.median(cosine_kernel_stream_list)}")
    #     print(f"Cosine (Without Stream): {cp.median(cosine_kernel_list)}")
    #     print(f"L2: {cp.median(l2_kernel_list)}")
    #     print(f"Dot: {cp.median(dot_kernel_list)}")
    #     print(f"Manhattan: {cp.median(manhattan_kernel_list)}")
    #     print("----------------------------------------")
    #     print("Differences in Speed (Positive means API is faster than Kernel)")
    #     print(f"Cosine Difference: {cp.median(cosine_kernel_list) - cp.median(cosine_api_list)}")
    #     print(f"Cosine Difference (Streams): {cp.median(cosine_kernel_stream_list) - cp.median(cosine_api_stream_list)}")
    #     print(f"L2 Difference: {cp.median(l2_kernel_list) - cp.median(l2_api_list)}")
    #     print(f"Dot Difference: {cp.median(dot_kernel_list) - cp.median(dot_api_list)}")
    #     print(f"Manhattan Difference: {cp.median(manhattan_kernel_list) - cp.median(manhattan_api_list)}")
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
    #     print(f"Cosine CPU: {cp.median(cosine_cpu)}")
    #     print(f"Cosine GPU: {cp.median(cosine_gpu)}")
    #     print(f"Cosine CPU - GPU: {(cp.median(cosine_cpu) - cp.median(cosine_gpu)).item()}")
    #     print(f"L2 CPU: {cp.median(l2_cpu)}")
    #     print(f"L2 GPU: {cp.median(l2_gpu)}")
    #     print(f"L2 CPU - GPU: {(cp.median(l2_cpu) - cp.median(l2_gpu)).item()}")
    #     print(f"Dot CPU: {cp.median(dot_cpu)}")
    #     print(f"Dot GPU: {cp.median(dot_gpu)}")
    #     print(f"Dot CPU - GPU: {(cp.median(dot_cpu) - cp.median(dot_gpu)).item()}")
    #     print(f"Manhattan CPU: {cp.median(cosine_cpu)}")
    #     print(f"Manhattan GPU: {cp.median(manhattan_gpu)}")
    #     print(f"Manhattan CPU - GPU: {(cp.median(manhattan_cpu) - cp.median(manhattan_gpu)).item()}")
    #     print("----------------------------------------")
    
    ### Test KNN
    ### Test KMeans
    
    ### Test Ann
    # test_our_ann()
    test_our_ann_IVFPQ()