"""
Memory Leak Diagnostic Script
"""

import torch
import gc
import sys


def get_tensor_memory():
    """Find all CUDA tensors in memory"""
    total_size = 0
    tensor_info = []
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size = obj.element_size() * obj.nelement()
                if size > 10 * 1024 * 1024:  # Only show tensors > 10 MB
                    tensor_info.append({
                        'shape': tuple(obj.shape),
                        'dtype': obj.dtype,
                        'size_mb': size / (1024**2),
                        'requires_grad': obj.requires_grad
                    })
                total_size += size
        except:
            pass
    
    return total_size, tensor_info


def print_memory_summary():
    """Print detailed memory summary"""
    print("\n" + "="*80)
    print("CUDA MEMORY SUMMARY")
    print("="*80)
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Allocated:     {allocated:.2f} GB")
    print(f"Reserved:      {reserved:.2f} GB")
    print(f"Max Allocated: {max_allocated:.2f} GB")
    print(f"Free:          {reserved - allocated:.2f} GB (reserved but not allocated)")
    
    # Get tensor info
    total_size, tensors = get_tensor_memory()
    print(f"\nTotal tensor size: {total_size / 1e9:.2f} GB")
    print(f"Number of large tensors (>10MB): {len(tensors)}")
    
    if tensors:
        print("\nLargest tensors:")
        sorted_tensors = sorted(tensors, key=lambda x: x['size_mb'], reverse=True)
        for i, t in enumerate(sorted_tensors[:10]):
            print(f"  {i+1}. {t['shape']} {t['dtype']} - {t['size_mb']:.1f} MB "
                  f"{'(grad)' if t['requires_grad'] else ''}")
    
    print("="*80 + "\n")


def aggressive_memory_cleanup():
    """Aggressively clean GPU memory"""
    print("Running aggressive memory cleanup...")
    
    # Clear Python cache
    gc.collect()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Clear all cached allocations
    torch.cuda.synchronize()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    print("Cleanup complete.\n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        
        print_memory_summary()
        aggressive_memory_cleanup()
        print_memory_summary()
    else:
        print("No CUDA device available!")
