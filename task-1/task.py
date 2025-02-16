import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
import scipy
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
subtract_square = cp.ElementwiseKernel('float64 x, float64 y', 
                                       'float64 z', 
                                       '''
                                       z = (x - y);
                                       z = z * z;                                   
                                       '''
                                       )

subtract_abs = cp.ElementwiseKernel('float64 x, float64 y',
                                    'float64 z',
                                    '''
                                    z = abs(x - y)
                                    ''')

sum_sqrt = cp.ReductionKernel('float64 x', 'float64 y', 'x', 'a + b', 'y = sqrt(a)', '0')

multiply = cp.ElementwiseKernel('float64 x, float64 y', 
                                'float64 z', 
                                '''z = x * y''')

_sum = cp.ReductionKernel('float64 x', 'float64 y', 'x', 'a + b', 'y = a', '0')

square = cp.ElementwiseKernel('float64 x', 'float64 y', '''y = x * x''')

divide = cp.ElementwiseKernel('float64 x, float64 y', 'float64 z', '''z = x / y''')

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

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K, distance_metric="l2", use_kernel = True):

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
            distance_manhattan(A, X, use_kernel) 
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

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_cosine(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_cosine(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))
    return end-start

def test_l2(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_l2(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = cp.linalg.norm(X - Y)
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))
    return end-start

def test_dot(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_dot(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = cp.dot(X, Y)
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))
    return end-start

def test_manhattan(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_manhattan(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = scipy.spatial.distance.cityblock(X.get(), Y.get())
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))
    return end-start

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    D = 2**15
    print(f"Dimension: {D}")
    print("----------------------------------------")
    print("Cosine Distance Test (Kernel)")
    cosine_kernel = test_cosine(D)
    print("Cosine Distance Test (API)")
    cosine_api = test_cosine(D, use_kernel=False)
    print("L2 Distance Test (Kernel)")
    l2_kernel = test_l2(D)
    print("L2 Distance Test (API)")
    l2_api = test_l2(D, use_kernel=False)
    print("Dot Distance Test (Kernel)")
    dot_kernel = test_dot(D)
    print("Dot Distance Test (API)")
    dot_api = test_dot(D, use_kernel=False)
    print("Manhattan Distance Test (Kernel)")
    manhattan_kernel = test_manhattan(D)
    print("Manhattan Distance Test (API)")
    manhattan_api = test_manhattan(D, use_kernel=False)
    print("----------------------------------------")
    print("Differences in Speed (Positive means API is faster than Kernel)")
    print(f"Cosine Difference: {cosine_kernel-cosine_api}")
    print(f"L2 Difference: {l2_kernel-l2_api}")
    print(f"Dot Difference: {dot_kernel-dot_api}")
    print(f"Manhattan Difference: {manhattan_kernel-manhattan_api}")