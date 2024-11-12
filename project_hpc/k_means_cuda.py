"""
k-Means Clustering
https://en.wikipedia.org/wiki/K-means_clustering
"""

import numpy as np
from numba import cuda
import math

@cuda.jit
def calculate_distances_gpu(X, centroids, distances):
    # Shared memory for centroids and data points
    shared_X = cuda.shared.array((32, 32), dtype=np.float32)
    shared_centroids = cuda.shared.array((32, 32), dtype=np.float32)

    i, j = cuda.grid(2)  # Thread indices

    # Load X and centroids into shared memory
    if i < X.shape[0] and j < centroids.shape[0]:
        for d in range(X.shape[1]):
            if cuda.threadIdx.x < 32 and cuda.threadIdx.y < 32:
                shared_X[cuda.threadIdx.x, d] = X[i, d]
                shared_centroids[cuda.threadIdx.y, d] = centroids[j, d]
        cuda.syncthreads()  # Synchronize to ensure shared memory is fully loaded

        # Compute the distance using shared memory
        dist = 0.0
        for d in range(X.shape[1]):
            diff = shared_X[cuda.threadIdx.x, d] - shared_centroids[cuda.threadIdx.y, d]
            dist += diff * diff
        distances[i, j] = math.sqrt(dist)  # Store the computed distance

@cuda.jit
def calculate_centroids_gpu(X, labels, centroids, n_points_per_cluster):
    # Shared memory for intermediate centroid sums
    shared_centroids = cuda.shared.array((32, 32), dtype=np.float32)

    i = cuda.grid(1)  # Thread index

    if i < X.shape[0]:  # Ensure we are within bounds
        cluster_id = labels[i]

        # Accumulate sums in shared memory
        for d in range(X.shape[1]):
            cuda.atomic.add(shared_centroids, (cluster_id, d), X[i, d])

        # Synchronize threads to make sure all sums are accumulated
        cuda.syncthreads()

        # Copy the shared memory values to the global centroids and count arrays
        if cuda.threadIdx.x == 0:
            for d in range(X.shape[1]):
                cuda.atomic.add(centroids, (cluster_id, d), shared_centroids[cluster_id, d])
            cuda.atomic.add(n_points_per_cluster, cluster_id, 1)

class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, random_state: int = None):
        """
        Initialize K-Means Clustering

        Parameters:
        - n_clusters: Number of clusters
        - max_iter: Maximum iterations
        - random_state: Seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        # Attributes to be set during fitting
        self.centroids = None
        self.labels = None


    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit the K-Means algorithm

        Parameters:
        - X: Input data array

        Returns:
        - Self
        """

        # print("Debugging KMeans (KM) - fit")
        # print("KM.fit.1")

        # Randomly initialize centroids
        if self.random_state is not None:
            np.random.seed(self.random_state)

        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        # print("KM.fit.2")

        # Allocate device memory
        d_X = cuda.to_device(X)
        d_centroids = cuda.to_device(self.centroids)
        d_distances = cuda.device_array((X.shape[0], self.n_clusters), dtype=np.float32)
        d_labels = cuda.device_array(X.shape[0], dtype=np.int32)
        n_points_per_cluster = cuda.device_array(self.n_clusters, dtype=np.int32)

        # print("KM.fit.3")

        threads_per_block = (32, 32)  # Optimized for shared memory
        blocks_per_grid_x = (X.shape[0] + (threads_per_block[0] - 1)) // threads_per_block[0]
        blocks_per_grid_y = (self.n_clusters + (threads_per_block[1] - 1)) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        for _ in range(self.max_iter):

            # Calculate distances
            calculate_distances_gpu[blocks_per_grid, threads_per_block](
                d_X, d_centroids, d_distances
            )
            cuda.synchronize()  # Synchronize all threads in the block to ensure all distances are computed

            # print("KM.fit.6")

            # Update labels
            distances = d_distances.copy_to_host()    # Copy distances back to the host
            labels = np.argmin(distances, axis=1)     # Get labels by finding the minimum distance

            # print("KM.fit.7")

            # Update centroids on GPU
            calculate_centroids_gpu[blocks_per_grid, threads_per_block](
                d_X, d_labels, d_centroids, n_points_per_cluster
            )
            cuda.synchronize()  # Synchronize all threads in the block to ensure all centroids are updated

            # print("KM.fit.8")

            # Copy centroids back to CPU to check for convergence
            counts = n_points_per_cluster.copy_to_host()
            counts[counts == 0] = 1   #1e-10  # Avoid division by zero, at least 1 per cluster
            new_centroids = d_centroids.copy_to_host() / counts[:, None]

            # Ensure centroids are valid before casting
            new_centroids = np.nan_to_num(new_centroids, nan=0.0, posinf=255, neginf=0)
            new_centroids = new_centroids.astype('uint8')

            # print("KM.fit.9")

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters:
        - X: Input data array

        Returns:
        - Cluster labels
        """

        # print("Debugging KMeans (KM) - predict")
        # print("KM.prd.1")

        # Allocate device memory
        d_X = cuda.to_device(X)
        d_centroids = cuda.to_device(self.centroids)
        d_distances = cuda.device_array((X.shape[0], self.n_clusters), dtype=np.float32)

        # print("KM.prd.2")

        threads_per_block = (32, 32)
        blocks_per_grid_x = (X.shape[0] + (threads_per_block[0] - 1)) // threads_per_block[0]
        blocks_per_grid_y = (self.n_clusters + (threads_per_block[1] - 1)) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # print("KM.prd.3")

        # Calculate distances
        calculate_distances_gpu[blocks_per_grid, threads_per_block](
            d_X, d_centroids, d_distances
        )
        cuda.synchronize()  # Ensure the GPU has finished processing before moving to the next steps

        # print("KM.prd.4")

        # Copy the distances back to host memory
        distances = d_distances.copy_to_host()

        # print("KM.prd.5")

        # Find the index of the minimum distance for each point (closest centroid)
        labels = np.argmin(distances, axis=1)

        return labels