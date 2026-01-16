#!/usr/bin/env python
"""
Test script for calculating FLOPs and FPS of ODTrack model
"""

import importlib
import time

import torch
from thop import profile
from thop import clever_format
from lib.utils.misc import NestedTensor


def get_data(bs, sz):
    """
    Generate dummy input data for testing
    """
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


def evaluate_model_flops_and_speed(model, template_list, search_list, device="cuda"):
    """
    Evaluate model FLOPs and speed (FPS)
    """
    # Calculate FLOPs
    print("Calculating FLOPs...")
    try:
        # Pass inputs as lists based on the model requirement
        macs, params = profile(model, inputs=(template_list, search_list),
                               custom_ops=None, verbose=False)
        flops = 2 * macs  # MACs ≈ FLOPs/2, so multiply by 2 for FLOPs
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        
        print(f'Model FLOPs: {flops_formatted}')
        print(f'Model Parameters: {params_formatted}')
    except Exception as e:
        print(f"Error during FLOPs calculation: {e}")
        return None, None, None, None
    
    # Speed test
    print("\nTesting speed...")
    T_w = 100  # Warmup iterations
    T_t = 500  # Timing iterations
    
    # Move to device if available
    if torch.cuda.is_available() and device == "cuda":
        model = model.cuda()
        # Ensure all inputs are moved to CUDA
        template_list = [t.cuda() if not t.is_cuda else t for t in template_list] 
        search_list = [s.cuda() if not s.is_cuda else s for s in search_list]
        torch.cuda.synchronize()
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i in range(T_w):
            _ = model(template_list, search_list)
    
    # Timing
    torch.cuda.synchronize() if torch.cuda.is_available() and device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for i in range(T_t):
            _ = model(template_list, search_list)
    torch.cuda.synchronize() if torch.cuda.is_available() and device == "cuda" else None
    end = time.time()
    
    avg_time = (end - start) / T_t
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    return flops, params, avg_time, fps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Directly specify the script and config here
    script_name = "odtrack"
    config_name = "baseline"  # You can change this to other configs like "baseline_got", etc.
    
    # Update configuration
    yaml_fname = f'experiments/{script_name}/{config_name}.yaml'
    config_module = importlib.import_module(f'lib.config.{script_name}.config')
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    
    # Set up model
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    
    if script_name == "odtrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_odtrack
        model = model_constructor(cfg, training=False)
        
        # Create dummy inputs - based on profile_model.py, template can be a list
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        
        # Based on the profile_model.py, inputs should be lists
        template_list = [template]  # Create a list of templates
        search_list = [search]      # Create a list of searches
        
        # Move model and inputs to device early
        if torch.cuda.is_available() and device == "cuda":
            model = model.cuda()
            template_list = [t.cuda() for t in template_list]
            search_list = [s.cuda() for s in search_list]
        
        print(f"Model: ODTrack ({config_name})")
        print(f"Template size: {z_sz}x{z_sz}")
        print(f"Search size: {x_sz}x{x_sz}")
        
        # Calculate FLOPs and FPS
        flops, params, avg_time, fps = evaluate_model_flops_and_speed(model, template_list, search_list, device)
        
        if flops is not None and params is not None:
            # Print summary
            print("\n" + "="*50)
            print("SUMMARY:")
            print(f"Model: ODTrack ({config_name})")
            print(f"FLOPs: {clever_format([2 * flops], '%.3f')[0]}")  # Convert MACs to FLOPs
            print(f"Parameters: {clever_format([params], '%.3f')[0]}")
            print(f"Average inference time: {avg_time * 1000:.2f} ms")
            print(f"FPS: {fps:.2f}")
            print("="*50)
        else:
            print("Could not calculate FLOPs and FPS due to error")
        
    else:
        print(f"Script {script_name} not recognized. Only 'odtrack' is supported.")
        return


if __name__ == "__main__":
    main()