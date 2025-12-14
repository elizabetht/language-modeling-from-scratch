"""
Simple PyTorch Device Example
"""

import torch


def main():
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"PyTorch Device: {device}")
    print(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
    # Create a tensor
    x = torch.ones(4, 8)
    print(f"\nTensor device: {x.device}")


if __name__ == "__main__":
    main()