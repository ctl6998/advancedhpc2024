import os
import time
import numpy as np
from numba import cuda
from PIL import Image  # Import PIL for saving as JPEG

# Get image
image_path = 'image2.jpg'  
image = Image.open(image_path).convert('RGB')  # Read the image properly using PIL
print(f"Processing images:")
print(f"Image width:",image.width)
print(f"Image width:",image.height)
pixels = np.array(image)
pixel_count = image.width * image.height  # Total number of pixels
pixels = pixels.reshape(-1, 3)  # Reshape to (pixel_count, 3) for RGB

print(f"Processing pixels.")
print(pixels)


### RGB to Gray: CPU
print("===================CPU PROCESSING===================")
start_cpu = time.time()
host_output_cpu = np.zeros((pixel_count, 3), dtype=np.uint8)  # Adjust output to store RGB values
for i, (r, g, b) in enumerate(pixels):
    # Convert RGB values to int to prevent overflow
    r = int(r)
    g = int(g)
    b = int(b)
    # Using the average method for grayscale conversion
    g_value = (r + g + b) // 3
    host_output_cpu[i] = (g_value, g_value, g_value)  # Set all channels to the grayscale value
end_cpu = time.time()
print(host_output_cpu)


### RGB to Gray: GPU
print("===================GPU PROCESSING===================")
## 1. CPU feeds data to GPU: Allocate memory
dev_input = cuda.to_device(pixels) #Must be (Nx3) matrix, with N=number of pixel and 3 is channel
dev_output = cuda.device_array((pixel_count, 3), dtype=np.uint8)  # Adjust output to store RGB values

## 2. CPU asks GPU to process
block_size = 64
grid_size = pixel_count // block_size

## 3. GPU processing with Kernel:
@cuda.jit
def grayscale_kernel(src, dst):
    # Calculate the index for each thread
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < src.shape[0]:
        # Compute grayscale value using average method
        g = (src[idx, 0] + src[idx, 1] + src[idx, 2]) // 3
        dst[idx, 0] = dst[idx, 1] = dst[idx, 2] = g

grayscale_kernel[grid_size, block_size](dev_input, dev_output)

# 4. GPU copy results to CPU
host_output_gpu = dev_output.copy_to_host()
print(host_output_gpu)

# Print processing times
print("===================COMPARISON===================")
cpu_time = end_cpu - start_cpu
gpu_time = time.time() - end_cpu
print(f'CPU Time: {cpu_time:.4f} seconds')
print(f'GPU Time: {gpu_time:.4f} seconds')
print(f'Speedup: {cpu_time / gpu_time:.2f}x')

# Saving resutls
print("===================SAVING===================")
def save_grayscale_image(output_array, output_path, width, height):
    ## Output from CPU and GPU are all (N,3) matrix, with each row a pixel
    gray_image = Image.new('RGB', (width, height))
    gray_image.putdata([tuple(row) for row in output_array])  # Use all channels for grayscale
    gray_image.save(output_path)

cpu_output_path = os.path.splitext(image_path)[0] + '_gray_cpu.jpg'
save_grayscale_image(host_output_cpu, cpu_output_path, image.width, image.height)

gpu_output_path = os.path.splitext(image_path)[0] + '_gray_gpu.jpg'
save_grayscale_image(host_output_gpu, gpu_output_path, image.width, image.height)

print(f"Saving results completed")