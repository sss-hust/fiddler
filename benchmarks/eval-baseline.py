# fix numpy in colab
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy
import os
import sys
import argparse
import logging

sys.path.append("mixtral-offloading")


def main():
    os.chdir("mixtral_offloading")

    if args.framework == 'mixtral-offloading':
        logging.info('Using mixtral-offloading')
        model = init_mixtral_offload()
    elif args.framework == 'deepspeed-mii':
        logging.info('Using deepspeed-mii')
        model = init_deepspeed_mii()
    else:
        raise ValueError(f'Unknown framework: {args.framework}')

    eval(model)


def init_deepspeed_mii():
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig

    model_id = "mistralai/Mixtral-8x7B-v0.1"
    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            }
        },
        "train_micro_batch_size_per_gpu": 1,
    }

    hfdsc = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16)

    deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model


def init_mixtral_offload():
    from hqq.core.quantize import BaseQuantizeConfig
    from mixtral_offloading.src.build_model import OffloadConfig, QuantConfig, build_model

    quantized = False

    if not quantized:
        state_path = "/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1"
        model_name = "/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1"
    else:
        state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
        model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    config = AutoConfig.from_pretrained(model_name)

    device = torch.device("cuda:0")

    ##### Change this to 5 if you have only 12 GB of GPU VRAM #####
    # offload_per_layer = 4
    offload_per_layer = 5
    ###############################################################

    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )

    if quantized:
        quant_config = QuantConfig(
            ffn_config=ffn_config,
            attn_config=attn_config)
    else:
        quant_config = None

    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )
    return model


def eval(model):
    import random
    import json
    import time

    device = torch.device("cuda:0")

    path_json = '/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1/ShareGPT_V3_unfiltered_cleaned_split.json'
    with open(path_json, 'r') as f:
        data = json.load(f)
    texts = []
    for d in data:
        if len(d['conversations']) == 0:
            continue
        # the input of the first round
        texts.append(' '.join(d['conversations'][0]['value'].split()))

    logging.info(f'n of input {len(texts)}')
    random.seed(0)
    random.shuffle(texts)

    n_sample = 3

    model_name = "/home/yangfu/model/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for input_token in [16, 32, 64, 128]:
        for output_token in [16]:
            idx_text = 0
            time_sum = 0
            num_tokens = 0
            logging.info(
                f'evaluating -- input_token: {input_token}, output_token: {output_token}')
            prefill_time_sum = 0
            decode_time_sum = 0
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                # print(f'input text: {text.split()[:input_token]}')
                input_ids = tokenizer.encode(
                    text, return_tensors='pt').to(device)
                # 1. Prefill 阶段 (处理 Prompt)
                torch.cuda.synchronize()
                t_start = time.time()
                
                # 执行第一次前向传播，处理所有输入 Token
                with torch.no_grad():
                    outputs = model(input_ids[:, :input_token], use_cache=True)
                
                torch.cuda.synchronize()
                t_prefill_end = time.time()
                
                # 记录 Prefill 时间
                prefill_cur = t_prefill_end - t_start
                prefill_time_sum += prefill_cur
                
                # 准备进入 Decode 阶段
                past_key_values = outputs.past_key_values
                # 取最后一个 Token 的 logits 并使用 Greedy Search 选择下一个 Token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                # 2. Decode 阶段 (逐个生成 Token)
                # 我们需要总共生成 output_token 个新 Token
                # 第 1 个已经在 Prefill 阶段计算出了 Logits，所以只需再循环 output_token - 1 次
                
                torch.cuda.synchronize()
                t_decode_start = time.time()
                
                with torch.no_grad():
                    for _ in range(output_token - 1):
                        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                        past_key_values = outputs.past_key_values
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                torch.cuda.synchronize()
                t_decode_end = time.time()
                
                # 记录 Decode 时间
                decode_cur = t_decode_end - t_decode_start
                decode_time_sum += decode_cur
                
                # 更新总时间 (为了兼容原有逻辑)
                time_sum += (prefill_cur + decode_cur)
                # count the number of tokens in the output
                # num_tokens += result["sequences"].shape[1]
                # print(f'output text: {tokenizer.decode(result["sequences"][0])}')

            logging.info(
                f'*******************\n'
                f'input_token: {input_token}, output_token: {output_token}, '
                # === 修改 C: 输出 Prefill 和 Decode 时间 ===
                f'prefill_time: {prefill_time_sum / n_sample:.4f}, '
                f'decode_time: {decode_time_sum / n_sample:.4f}, '
                f'token/s: {output_token / ((prefill_time_sum + decode_time_sum) / n_sample):.2f}\n'
                # =========================================
                f'*******************\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quantized', type=bool, default=False,
        help='Whether to use quantized model in mixtral-offloading.'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default='mixtral-offloading',
        choices=[
            'mixtral-offloading',
            'deepspeed-mii'],
        help='Which framework to use for evaluation.')

    args = parser.parse_args()

    # save log to file
    logging.basicConfig(filename='eval.log', level=logging.INFO)
    main()
