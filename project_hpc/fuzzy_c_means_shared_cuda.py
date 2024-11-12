"""
Fixed Fuzzy C-Means clustering implementation using CUDA with shared memory optimization.
"""

import numpy as np
from numba import cuda
import math
from typing import Tuple
import time

BLOCK_SIZE = 256
MAX_CLUSTERS = 32

@cuda.jit
def calculate_memberships_kernel(data, centroids, memberships, m):
    # Shared memory for centroids
    shared_centroids = cuda.shared.array(shape=(MAX_CLUSTERS, 3), dtype=np.float32)
    
    # Thread and block info
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    n_clusters = centroids.shape[0]
    
    # Load centroids into shared memory
    if tx < n_clusters:
        for j in range(data.shape[1]):
            shared_centroids[tx, j] = centroids[tx, j]
    cuda.syncthreads()
    
    # Calculate global thread index
    idx = bx * BLOCK_SIZE + tx
    if idx >= data.shape[0]:
        return
        
    # Calculate distances and update memberships
    distances = cuda.local.array(shape=(MAX_CLUSTERS,), dtype=np.float32)
    min_dist = 1e10
    min_idx = 0
    
    # Calculate distances
    for j in range(n_clusters):
        dist = 0.0
        for k in range(data.shape[1]):
            diff = data[idx, k] - shared_centroids[j, k]
            dist += diff * diff
        distances[j] = math.sqrt(dist) # distance
        if dist < min_dist:
            min_dist = dist
            min_idx = j
    
    # Handle the case where a point is exactly on a centroid
    if min_dist < 1e-10:
        for j in range(n_clusters):
            memberships[idx, j] = 1.0 if j == min_idx else 0.0
        return
    
    # Calculate memberships
    sum_inv_dist = 0.0
    power = 2.0 / (m - 1.0)
    
    for j in range(n_clusters):
        sum_inv_dist += (1.0 / distances[j]) ** power
        
    for j in range(n_clusters):
        memberships[idx, j] = (1.0 / distances[j]) ** power / sum_inv_dist

@cuda.jit
def update_centroids_kernel(data, memberships, centroids, m):
    cluster_idx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    
    if cluster_idx >= centroids.shape[0]:
        return
        
    # Shared memory for partial sums and denominators
    shared_nums = cuda.shared.array(shape=(BLOCK_SIZE, 3), dtype=np.float32)
    shared_dens = cuda.shared.array(shape=(BLOCK_SIZE,), dtype=np.float32)
    
    # Initialize shared memory
    for j in range(3):
        shared_nums[tx, j] = 0.0
    shared_dens[tx] = 0.0
    cuda.syncthreads()
    
    # Process data points in chunks
    for i in range(tx, data.shape[0], BLOCK_SIZE):
        if i < data.shape[0]:
            membership = memberships[i, cluster_idx] ** m
            shared_dens[tx] += membership
            
            for j in range(3):
                shared_nums[tx, j] += membership * data[i, j]
    
    cuda.syncthreads()
    
    # Reduction in shared memory
    s = BLOCK_SIZE // 2
    while s > 0:
        if tx < s:
            shared_dens[tx] += shared_dens[tx + s]
            for j in range(3):
                shared_nums[tx, j] += shared_nums[tx + s, j]
        cuda.syncthreads()
        s //= 2
    
    # Update centroid
    if tx == 0 and shared_dens[0] > 0:
        for j in range(3):
            centroids[cluster_idx, j] = shared_nums[0, j] / shared_dens[0]

class OptimizedFuzzyCMeans:
    def __init__(self, n_clusters, max_iter, tol: float = 1e-4, m: float = 2.0):
        if n_clusters > MAX_CLUSTERS:
            raise ValueError(f"Number of clusters cannot exceed {MAX_CLUSTERS}")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.m = m
        self.centroids = None
        self.memberships = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx].copy()
    
    def fit(self, X: np.ndarray) -> 'OptimizedFuzzyCMeans':
        # Normalize data
        X = X.astype(np.float32) / 255.0
        
        # Initialize centroids and memberships
        self.centroids = self._initialize_centroids(X)
        self.memberships = np.random.rand(X.shape[0], self.n_clusters).astype(np.float32)
        self.memberships /= self.memberships.sum(axis=1, keepdims=True)
        
        # CPU to GPU
        X_gpu = cuda.to_device(X)
        centroids_gpu = cuda.to_device(self.centroids)
        memberships_gpu = cuda.to_device(self.memberships)
        prev_centroids = np.zeros_like(self.centroids)
        
        # CPU ask GPU
        blocks_membership = (X.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # GPU processing
        for iteration in range(self.max_iter):
            # Store previous centroids
            prev_centroids[:] = cuda.to_device(centroids_gpu).copy_to_host()
            
            # Update memberships
            calculate_memberships_kernel[blocks_membership, BLOCK_SIZE](
                X_gpu, centroids_gpu, memberships_gpu, self.m)
            
            # Update centroids
            update_centroids_kernel[self.n_clusters, BLOCK_SIZE](
                X_gpu, memberships_gpu, centroids_gpu, self.m)
            
            # Check convergence
            current_centroids = cuda.to_device(centroids_gpu).copy_to_host()
            centroid_shift = np.max(np.abs(current_centroids - prev_centroids))
            
            if centroid_shift < self.tol:
                break
                
            self.n_iter_ = iteration + 1
        
        # GPU to CPU
        self.centroids = centroids_gpu.copy_to_host()
        self.memberships = memberships_gpu.copy_to_host()
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict cluster memberships and get quantized colors"""
        X = X.astype(np.float32) / 255.0
        
        X_gpu = cuda.to_device(X)
        centroids_gpu = cuda.to_device(self.centroids)
        memberships_gpu = cuda.to_device(np.zeros((X.shape[0], self.n_clusters), dtype=np.float32))
        
        blocks = (X.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        calculate_memberships_kernel[blocks, BLOCK_SIZE](
            X_gpu, centroids_gpu, memberships_gpu, self.m)
        
        memberships = memberships_gpu.copy_to_host()
        quantized = np.dot(memberships, self.centroids)
        
        return (quantized * 255).astype(np.uint8), memberships