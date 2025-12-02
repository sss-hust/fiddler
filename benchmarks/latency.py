"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys

sys.path.append("../src")
from fiddler import FiddlerMixtral

import torch.cuda.nvtx as nvtx
import numpy as np

class NVTXProfiler:
    def __init__(self):
        self.first_forward = True
        self.decode_step = 1

        # 记录数据
        self.records = {
            "prefill_ms": None,
            "decode_ms": [],   # 每个 decode token 的耗时
        }

        self.forward_start_ts = None

    def pre_hook(self, module, inputs):
        torch.cuda.synchronize()
        self.forward_start_ts = time.perf_counter()

        if self.first_forward:
            nvtx.range_push("Prefill")
        else:
            nvtx.range_push(f"Decode_{self.decode_step}")

    def post_hook(self, module, inputs, outputs):
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self.forward_start_ts) * 1000  # ms

        if self.first_forward:
            self.records["prefill_ms"] = elapsed
            nvtx.range_pop()  # end Prefill
            self.first_forward = False
        else:
            self.records["decode_ms"].append(elapsed)
            nvtx.range_pop()  # end Decode_n
            self.decode_step += 1

    def summary(self):

        decode_times = np.array(self.records["decode_ms"], dtype=float)
        # 如果没有 decode tokens（极少数情况）
        if len(decode_times) == 0:
            return {
                "prefill_ms": self.records["prefill_ms"],
                "decode_avg_ms": None,
                "decode_p99_ms": None,
                "decode_jitter_cv": None,
                "decode_total_ms": None,
                "decode_throughput_tokens_per_s": None,
                "decode_steps": 0,
            }

        decode_avg = decode_times.mean()
        decode_p99 = np.percentile(decode_times, 99)
        decode_jitter_cv = decode_times.std() / decode_times.mean()  # Coefficient of Variation
        decode_total_ms = decode_times.sum()
        decode_tp = len(decode_times) / (decode_total_ms / 1000.0)     # tokens / sec

        return {
            "prefill_ms": self.records["prefill_ms"],          # Prefill 阶段总耗时
            "decode_avg_ms": decode_avg,                       # 平均 TPOT
            "decode_p99_ms": decode_p99,                       # P99 TPOT
            "decode_jitter_cv": decode_jitter_cv,              # 变异系数（衡量 jitter）
            "decode_total_ms": decode_total_ms,                # Decode 总耗时
            "decode_throughput_tokens_per_s": decode_tp,       # 吞吐率
            "decode_steps": len(decode_times),                 # token 数量
        }

def attach_nvtx_hooks(model):
    profiler = NVTXProfiler()
    model._nvtx_pre_hook = model.register_forward_pre_hook(profiler.pre_hook)
    model._nvtx_post_hook = model.register_forward_hook(profiler.post_hook)

    return profiler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference.",
    )
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")

    args = parser.parse_args()

    path_json = "/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1/ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    model = FiddlerMixtral(args)
    profiler = attach_nvtx_hooks(model)
    n_sample = 10

    for input_token in [128]:
        for output_token in [8]:
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                prefill_time, decode_time, hit_rate = model.generate(
                    [text], output_token=output_token, input_token=input_token
                )
                prefill_time_sum += prefill_time
                decode_time_sum += decode_time
                hit_rate_sum += hit_rate
            # write to file
            with open("latency.txt", "a") as f:
                f.write(
                    json.dumps(profiler.summary(), indent=4)
                )
