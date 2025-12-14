"""
Simple PyTorch Device Example
"""

import torch


def device():
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def memory():
    memory_allocated = torch.cuda.memory_allocated()
    
    # Create a tensor
    x = torch.ones(32, 32, device="cuda:0")
    new_memory_allocated = torch.cuda.memory_allocated()
    memory_used = new_memory_allocated - memory_allocated
    print(f"Memory_Allocated before tensor: {memory_allocated}")
    print(f"Memory_Allocated after tensor: {new_memory_allocated}")
    print(f"Memory used by tensor: {memory_used}")
    
    # Actual tensor size
    tensor_size = 32 * 32 * 4  # 32x32 elements × 4 bytes per float32
    print(f"Expected tensor size: {tensor_size} bytes")
    print(f"Allocation ratio: {memory_used / tensor_size}x")
    
    # Note: PyTorch's memory allocator behavior varies by version
    # Older versions allocated 2x, newer versions may allocate exactly what's needed

if __name__ == "__main__":
    # device()
    memory()