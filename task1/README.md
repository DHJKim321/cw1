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

**What is ANN algorithm:**


**Pesudo code:**
```
1. Use KMeans to cluster the data into K clusters
2. In each query, find the nearest K1 cluster center as the approximate nearest neighbor
3. Use KNN to find the nearest K2 neighbor from the K1 cluster centers
4. Merge K1 * K2 vectors and find top K neighbors
```

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  X: A specified vector
-  K: Top K

**Output:**
-  Result[K]: The top K nearest vectors ID (index of the vector in A)



