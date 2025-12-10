"""Microbenchmarking for CPU offloading with Nsight Profiling"""

import argparse
import json
import os
import random
import sys
import time
import functools
import torch
import torch.cuda.nvtx as nvtx
import numpy as np

# 确保能找到 fiddler 源码
sys.path.append("../src")
try:
    from fiddler import FiddlerMixtral
except ImportError:
    pass # 允许在没有库的环境下做语法检查

class NsightProfiler:
    """
    非侵入式 Nsight 性能分析注入工具
    """
    
    # --- 新增内部类：用于追踪 Prefill/Decode 阶段 ---
    class GenerationPhaseTracker:
        def __init__(self):
            self.step = 0
            
        def reset(self):
            self.step = 0

        def pre_hook(self, module, input):
            # 强制同步，确保时间准确
            torch.cuda.synchronize()
            
            # 第0次调用通常是 Prefill
            if self.step == 0:
                nvtx.range_push("Phase: Prefill")
            else:
                nvtx.range_push(f"Phase: Decode_{self.step}")

        def post_hook(self, module, input, output):
            torch.cuda.synchronize()
            nvtx.range_pop() # 结束当前 Phase
            self.step += 1

    @staticmethod
    def wrap_method(instance, method_name, tag_name=None):
        """
        劫持最外层的方法 (如 generate)，记录总耗时
        """
        if not hasattr(instance, method_name):
            print(f"Warning: Method {method_name} not found.")
            return

        original_method = getattr(instance, method_name)
        tag = tag_name or f"{instance.__class__.__name__}.{method_name}"

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            nvtx.range_push(f"[{tag}]")
            
            # 如果实例绑定了 phase_tracker，每次 generate 前重置计数器
            if hasattr(instance, "_nsight_phase_tracker"):
                instance._nsight_phase_tracker.reset()
            
            try:
                result = original_method(*args, **kwargs)
                return result
            finally:
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                nvtx.range_pop()
                print(f"⏱️  [Time] {tag}: {(end_time - start_time) * 1000:.2f} ms")

        setattr(instance, method_name, wrapper)
        print(f"✅ Instrumented method: {method_name}")

    @staticmethod
    def hook_generation_phases(torch_model, wrapper_instance=None):
        """
        【关键修复】给模型主干注册 Hook，用于区分 Prefill 和 Decode
        """
        tracker = NsightProfiler.GenerationPhaseTracker()
        
        # 注册到模型最顶层的 Forward 上
        torch_model.register_forward_pre_hook(tracker.pre_hook)
        torch_model.register_forward_hook(tracker.post_hook)
        
        # 将 tracker 绑定到 wrapper 实例上（方便 reset）
        if wrapper_instance:
            wrapper_instance._nsight_phase_tracker = tracker
            
        print(f"✅ Registered Phase Tracker (Prefill/Decode) on {torch_model.__class__.__name__}")

    @staticmethod
    def register_layer_hooks(torch_model):
        """
        给内部子层注册 Hook，查看细节 (Attention/MoE)
        """
        def pre_hook(module, input, name):
            nvtx.range_push(f"L:{name}")

        def post_hook(module, input, output, name):
            nvtx.range_pop()

        target_keywords = ("self_attn", "block_sparse_moe", "moe_layer", "experts", "mlp")

        count = 0
        for name, module in torch_model.named_modules():
            if not name: continue
            if any(k in name.lower() for k in target_keywords):
                module.register_forward_pre_hook(functools.partial(pre_hook, name=name))
                module.register_forward_hook(functools.partial(post_hook, name=name))
                count += 1
        
        print(f"✅ Registered Layer Hooks for {count} internal modules.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser.add_argument("--model", type=str, default="/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--cpu-offload", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_width", type=int, default=1)
    args = parser.parse_args()

    # 1. 加载数据 (同前)
    path_json = os.path.join(os.path.dirname(args.model), "ShareGPT_V3_unfiltered_cleaned_split.json")
    if not os.path.exists(path_json):
        path_json = "/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1/ShareGPT_V3_unfiltered_cleaned_split.json"
    
    texts = []
    if os.path.exists(path_json):
        try:
            with open(path_json, "r") as f:
                data = json.load(f)
            for d in data:
                if len(d.get("conversations", [])) > 0:
                    texts.append(" ".join(d["conversations"][0]["value"].split()))
        except: pass
    if not texts: texts = ["Test prompt " * 50] * 50

    random.seed(0)
    random.shuffle(texts)

    # 2. 初始化
    print(f"Initializing FiddlerMixtral...")
    model = FiddlerMixtral(args)

    # ==========================================
    # 注入 Nsight 监控 (已修复)
    # ==========================================
    print("\n--- Injecting Nsight Profiler Hooks ---")
    
    # 1. 劫持 generate，并在 generate 开始时重置计数器
    NsightProfiler.wrap_method(model, "generate", tag_name="Fiddler_Generate")
    
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        # 2. 【新增】注册 Phase Tracker，用于区分 Prefill 和 Decode
        # 它会数 Forward 调用的次数：第0次=Prefill, >0=Decode
        NsightProfiler.hook_generation_phases(model.model, wrapper_instance=model)
        
        # 3. 注册细节 Layer Hook
        NsightProfiler.register_layer_hooks(model.model)
    else:
        print("Warning: Could not find internal PyTorch model.")
    print("---------------------------------------\n")

    # 3. 开始测试
    print("Running Warmup...")
    try:
        model.generate(["Warmup"], output_token=1, input_token=32)
    except: pass
    print("Warmup Done.\n")

    n_sample = 5
    for input_token in [128]:
        for output_token in [8]:
            idx_text = 0
            for i in range(n_sample):
                text_input = []
                while True:
                    if idx_text >= len(texts): idx_text = 0
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        text_input = [text]
                        break
                
                print(f"Sample {i+1}/{n_sample}: Input ~{input_token}, Output {output_token}...")
                model.generate(text_input, output_token=output_token, input_token=input_token)

    print("\nBenchmark Finished.")