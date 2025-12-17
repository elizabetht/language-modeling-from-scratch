"""
Simple PyTorch Device Example
"""

import torch

def precision():
    # FP32
    # Sign - 1 bit
    # Exponent - 8 bits
    # Fraction/Mantissa - 23 bits
    x = torch.zeros(4,8)
    print("Default dtype:", x.dtype)
    print("Size =", x.size())
    print("Numel =", x.numel())
    print("Element size (bytes) =", x.element_size())
    print("Total size (bytes) =", x.numel() * x.element_size())

    # FP16
    # Sign - 1 bit
    # Exponent - 5 bits
    # Fraction/Mantissa - 10 bits
    x_half = torch.zeros(4,8, dtype=torch.float16)
    print("Float16 dtype:", x_half.dtype)
    print("Size =", x_half.size())
    print("Numel =", x_half.numel())
    print("Element size (bytes) =", x_half.element_size())
    print("Total size (bytes) =", x_half.numel() * x_half.element_size())
    x_half_exp = torch.tensor([1e-10], dtype=torch.float16)
    print("Float16 small value:", x_half_exp.item())

    # BF16
    # Sign - 1 bit
    # Exponent - 8 bits
    # Fraction/Mantissa - 7 bits
    x_bfloat16 = torch.zeros(4,8, dtype=torch.bfloat16)
    print("BFloat16 dtype:", x_bfloat16.dtype)
    print("Size =", x_bfloat16.size())
    print("Numel =", x_bfloat16.numel())
    print("Element size (bytes) =", x_bfloat16.element_size())
    print("Total size (bytes) =", x_bfloat16.numel() * x_bfloat16.element_size())
    x_bfloat16_exp = torch.tensor([1e-10], dtype=torch.bfloat16)
    print("BFloat16 small value:", x_bfloat16_exp.item())

    # FP4
    # Sign - 1 bit
    # Exponent - 3 bits
    # Fraction/Mantissa - 2 bits
    if hasattr(torch, 'float4'):
        x_float4 = torch.zeros(4,8, dtype=torch.float4)
        print("Float4 dtype:", x_float4.dtype)
        print("Size =", x_float4.size())
        print("Numel =", x_float4.numel())
        print("Element size (bytes) =", x_float4.element_size())
        print("Total size (bytes) =", x_float4.numel() * x_float4.element_size())
        x_float4_exp = torch.tensor([1e-2], dtype=torch.float4)
        print("Float4 small value:", x_float4_exp.item())
    else:
        print("Float4 not supported in this version of PyTorch.")

def device():
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Device {i} properties: {torch.cuda.get_device_properties(i)}")

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

def tensor_slicing():
    x = torch.tensor([[1,2,3],[4,5,6]])
    y = x[0]
    print("storage = same" if x.storage().data_ptr() == y.storage().data_ptr() else "storage = different")

def matmul():
    a = torch.ones(16,32)
    b = torch.ones(32,2)
    c = a @ b
    print("C shape:", c.size())
    assert c.size() == (16,2)

if __name__ == "__main__":
    # precision()
    # device()
    # memory()
    tensor_slicing()
    # matmul()