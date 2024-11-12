"""
Fuzzy Clustering (Fuzzy C Mean or Soft K-mean)
https://en.wikipedia.org/wiki/Fuzzy_clustering
"""

import numpy as np
from numba import cuda
import math
from typing import Tuple, Optional
import time

############################ KERNEL ############################
@cuda.jit
def normalize_memberships_kernel(memberships):
    """Normalize membership values so they sum to 1 for each data point"""
    idx = cuda.grid(1)
    if idx >= memberships.shape[0]:
        return
        
    # Calculate sum for normalization
    row_sum = 0.0
    for j in range(memberships.shape[1]):
        row_sum += memberships[idx, j]
        
    # Normalize
    if row_sum > 0:
        for j in range(memberships.shape[1]):
            memberships[idx, j] /= row_sum

@cuda.jit
def calculate_memberships_kernel(data, centroids, memberships, m):
    idx = cuda.grid(1)
    if idx >= data.shape[0]: # H*W
        return

    # For reach pixel, calculate distances to all centroids
    # wᵢⱼ = 1 / Σₖ (||xᵢ - cⱼ|| / ||xᵢ - cₖ||) ^ (2/(m-1))
    n_clusters = centroids.shape[0]
    distances = cuda.local.array(shape=(32,), dtype=np.float64)  # Max 32 clusters
    
    # For each pixel (xᵢ): ||xᵢ - cₖ||,j from [1,k]
    for j in range(n_clusters):
        dist = 0.0
        for k in range(data.shape[1]):
            diff = data[idx, k] - centroids[j, k]
            dist += diff * diff
        distances[j] = math.sqrt(dist) # Euclidean distance
    
    # Update distance for each cluster (color: j
    for j in range(n_clusters):
        if distances[j] <= 1e-10:  # Point is very close to centroid
            for k in range(n_clusters):
                memberships[idx, k] = 1.0 if k == j else 0.0
            return
            
        #  Σₖ (||xᵢ - cⱼ|| / ||xᵢ - cₖ||) ^ (2/(m-1))
        den = 0.0
        for k in range(n_clusters):
            if distances[k] <= 1e-10:
                den = float('inf')
                break
            den += (distances[j] / distances[k]) ** (2.0/(m-1.0))
            
        memberships[idx, j] = 1.0 / den

@cuda.jit
def update_centroids_kernel(data, memberships, centroids, m):

    # Apply a reduce technique to membership to comnpute the updated centroid based on all pixels
    # c_j = Σᵢ (wᵢⱼ)ᵐ xᵢ / Σᵢ (wᵢⱼ)ᵐ
    idx = cuda.grid(1)
    if idx >= centroids.shape[0]: 
        return
        
    num = cuda.local.array(shape=(3,), dtype=np.float64)  # For RGB
    den = 0.0
    
    for j in range(data.shape[1]):
        num[j] = 0.0
    
    for i in range(data.shape[0]):
        membership_m = memberships[i, idx] ** m # Σᵢ (wᵢⱼ)ᵐ
        den += membership_m
        
        for j in range(data.shape[1]):
            num[j] += membership_m * data[i, j] # Σᵢ (wᵢⱼ)ᵐ xᵢ
    
    if den > 0:
        for j in range(data.shape[1]):
            centroids[idx, j] = num[j] / den #c_j = Σᵢ (wᵢⱼ)ᵐ xᵢ / Σᵢ (wᵢⱼ)ᵐ

@cuda.jit
def calculate_cost_kernel(data, centroids, memberships, cost_array, m):
    
    # Reducer, scatter and gather pattern
    # J = Σᵢ Σⱼ (wᵢⱼ)ᵐ ||xᵢ - cⱼ||²
    idx = cuda.grid(1)
    if idx >= data.shape[0]:
        return
        
    cost = 0.0
    for j in range(centroids.shape[0]):
        dist = 0.0
        for k in range(data.shape[1]): # 3 - RGB channel
            diff = data[idx, k] - centroids[j, k]
            dist += diff * diff # ||xᵢ - cⱼ||²
        membership_m = memberships[idx, j] ** m #(wᵢⱼ)ᵐ
        cost += membership_m * dist   # J = Σᵢ Σⱼ (wᵢⱼ)ᵐ ||xᵢ - cⱼ||²
    
    cost_array[idx] = cost


################################################################

class FuzzyCMeans:
    def __init__(self, n_clusters: int = 8, max_iter: int = 10, tol: float = 1e-4, m: float = 2.0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.memberships = None
        self.n_iter_ = 0
        self.m = m
        
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Choose randomly 2 main colors to quantize the image
        This is defition of number of cluster (k)
        """
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx].copy()
    
    def fit(self, X: np.ndarray) -> 'FuzzyCMeans':
        """
        The training iteration:
        1. Initialize random cluster (centroid): c_j (j from [1,k]) |k: number of cluster, m: fuzzy level
        2. Caculate it's "distance" to each pixel (called membership value): wᵢⱼ = 1 / Σₖ (||xᵢ - cⱼ|| / ||xᵢ - cₖ||) ^(2/(m-1))
           Equivalent to @kernel: calculate_memberships_kernel
        3. Update the new cluster centroid with formula: c_j = Σᵢ (wᵢⱼ)ᵐ xᵢ / Σᵢ (wᵢⱼ)ᵐ
           Equivalent to @kernel: update_centroids_kernel
        4. Run iteration until reach max_iteration OR cluster centroid change < threshold (converge): J = Σᵢ Σⱼ (wᵢⱼ)ᵐ ||xᵢ - cⱼ||²
           Equivalent to @kernel: calculate_cost_kernel
        Returns:
            self: The fitted clusterer
        """
        # Normalize dRGB data
        X = X.astype(np.float64) / 255.0
        
        # Initialize centroids and memberships
        self.centroids = self._initialize_centroids(X)
        self.memberships = np.random.rand(X.shape[0], self.n_clusters).astype(np.float64)
        self.memberships /= self.memberships.sum(axis=1, keepdims=True)
        
        # CPU to GPU
        X_gpu = cuda.to_device(X)
        centroids_gpu = cuda.to_device(self.centroids)
        memberships_gpu = cuda.to_device(self.memberships)
        cost_gpu = cuda.to_device(np.zeros(X.shape[0], dtype=np.float64))
        
        # CPU ask GPU
        threads_per_block = 256
        blocks_per_grid = (X.shape[0] + threads_per_block - 1) // threads_per_block
        prev_cost = float('inf')
        
        # GPU Processing
        for iteration in range(self.max_iter):
            # 1. Update memberships/measure "distance"
            # Write result to memberships_gpu
            calculate_memberships_kernel[blocks_per_grid, threads_per_block](
                X_gpu, centroids_gpu, memberships_gpu, self.m)
            
            # Normalize memberships
            # normalize_memberships_kernel[blocks_per_grid, threads_per_block](
            #     memberships_gpu)
            
            # 2. Update centroids
            # Write resuylt to centroids_gpu
            centroid_blocks = (self.n_clusters + threads_per_block - 1) // threads_per_block
            update_centroids_kernel[centroid_blocks, threads_per_block](
                X_gpu, memberships_gpu, centroids_gpu, self.m)
            
            # 3. Calculate centroid change
            # Write result to cost_gpu
            calculate_cost_kernel[blocks_per_grid, threads_per_block](
                X_gpu, centroids_gpu, memberships_gpu, cost_gpu, self.m)
            
            # 4. Check convergence
            current_cost = cost_gpu.copy_to_host().sum()
            if abs(prev_cost - current_cost) < self.tol:
                break
                
            prev_cost = current_cost
            self.n_iter_ = iteration + 1
            print("Finish iteration ${iteration}")
        
        # GPU to CPU
        self.centroids = centroids_gpu.copy_to_host()
        self.memberships = memberships_gpu.copy_to_host()
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict cluster memberships and get quantized colors.
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple containing:
            - Quantized colors (n_samples, n_features)
            - Membership matrix (n_samples, n_clusters)
        """
        X = X.astype(np.float64) / 255.0
        
        X_gpu = cuda.to_device(X)
        centroids_gpu = cuda.to_device(self.centroids)
        memberships_gpu = cuda.to_device(np.zeros((X.shape[0], self.n_clusters), dtype=np.float64))
        
        threads_per_block = 256
        blocks_per_grid = (X.shape[0] + threads_per_block - 1) // threads_per_block
        calculate_memberships_kernel[blocks_per_grid, threads_per_block](
            X_gpu, centroids_gpu, memberships_gpu, self.m)
        

        memberships = memberships_gpu.copy_to_host()
        quantized = np.dot(memberships, self.centroids)
        
        # Scale back to [0, 255]
        return (quantized * 255).astype(np.uint8), memberships