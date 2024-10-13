import numba
from numba import cuda

# Detect and print all available GPUs
print("===========================")
print("Detecting GPUs")
cuda.detect()

print("===========================")
print("Extracting info from device with ID")
device_id = 0
device = cuda.select_device(device_id)
print(f"Selected GPU ID: {device.id}")
print(f"Selected GPU Name: {device.name.decode('utf-8')}")

# Print selected GPU information
print("===========================")
print(f"Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")
print(f"Core count for compute capability 8.9: {device.MULTIPROCESSOR_COUNT * 128}")

# Memory information
print("===========================")
memory_info = cuda.current_context().get_memory_info()
total_memory = memory_info[1]  
print(f"Memory info: {memory_info}")
print(f"Total Memory (in GB): {total_memory / (1024**3):.2f} GB")