# Task 1

## Part 1: KNN

### Part 1.1: Distance Functions

Implement distance functions for two vectors in cupy/torch/triton.  

**Input:**  
- D: dimension of the vector.  
- X[D], Y[D]: two vectors  

**Output:**  
- the distance between X and Y 

There are four distinct types of distance, so the implementation of four separate functions:

**Cosine distance:**:

$d(X, Y) = 1 - \frac{X \cdot Y}{\|X\| \|Y\|}$

**L2 distance:**:

$d(X, Y) = \sqrt{\sum_{i=1}^D (X_i - Y_i)^2}$

**Dot product:**:

$d(X, Y) = X \cdot Y$

**Manhattan(L1) distance:**:

$d(X, Y) = \sum_{i=1}^D |X_i - Y_i|$

### Part 1.2: Top-K with GPU

Identify the K nearest vectors within a set of vectors.

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  X: A specified vector
-  K: Top K

**Output:**
-  Result[K]: The top K nearest vectors ID (index of the vector in A)



## Part 2: KMeans and ANN

### Part 2.1: KMeans

**What is Kmeans algorithm:**

K-means clustering is an unsupervised learning algorithm that groups similar data points into K clusters. It works by iteratively assigning points to the nearest cluster center (centroid) and updating centroids based on the mean of assigned points until convergence.
In this task we only use L2 distance and cosine similarity.

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  K: number of clusters

**Output:**
-  Result[N]: cluster ID for each vector

### Part 2.2: ANN

Efficient **Approximate Nearest Neighbor (ANN)** search using **Inverted File Indexing + Product Quantization (IVFPQ)** — with three versions - Numpy(CPU), Cupy(GPU), and a customised kernel(GPU).

IVFPQ (Approximate Nearest Neighbor Search)


## How to Run
from ivfpq_numpy import our_ann_numpy

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  X: A specified vector
-  K: Top K

**Output:**
-  Result[K]: The top K nearest vectors ID (index of the vector in A)

# Overview #
top_k_indices = our_ann(N, D, A, X, K)

# Pipeline Summary #
1. Product Quantization (PQ)
def product_quantization_numpy(D, M, A):
    # Split vectors A ∈ ℝ^(N×D) into M sub-vectors of size D/M
    # Apply K-Means on each subspace (K=256)
    # Return: M codebooks and encoded labels per vector

2. IVF + PQ Index Construction
def ivfpq_index(N, D, A, num_clusters=100, M=8):
    # Run K-Means clustering on A → Assign to coarse clusters
    # Build inverted index: ivf_lists = {cluster_id: [data_indices]}
    # Apply PQ to compress A
    # Return: ivf_lists, cluster_centers, codebooks, encoded_data

3. IVFPQ Search
def search_ivfpq(X, ivf_lists, cluster_centers, codebooks, encoded_data, A, K):
    # 1. Find nearest coarse clusters to query X (IVF)
    # 2. Gather candidates from inverted lists
    # 3. Compute approximate PQ distance (batch decoding)
    # 4. Re-rank top candidates using exact L2 distance
    # Return: Top-K ANN indices

# Notes # 
Custom K-Means: our_kmeans_batch(...)
PQ splits high-dim vectors into subspaces for compression
Inverted index limits search scope
Final re-ranking improves accuracy

