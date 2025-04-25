from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import utils
import sys
# sys.path.append('/scratch/jiarui14/EM-CoT/Online-DPO-R1')
# import reward_labeling

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

@dataclass
class ScriptArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    alpha: Optional[float] = field(
        default=1e-3,
        metadata={"help": "Penalty weight alpha"}
    )
    beta: Optional[float] = field(
        default=2.0,
        metadata={"help": "Penalty weight beta"}
    )
    stage_1_samples: Optional[int] = field(
        default=8,
        metadata={"help": "Number of samples for stage 1 per example"}
    )
    stage_2_samples: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of samples for stage 2 per example"}
    )
    iter: Optional[int] = field(
        default=1,
        metadata={"help": "the iteration index"}
    )
    model_prefix: Optional[str] = field(
        default="Qwen7B",
        metadata={"help": "the prefix of the model"}
    )
    suffix: Optional[str] = field(
        default="",
        metadata={"help": "the suffix"}
    )
    num_collect_files: Optional[int] = field(
        default=8,
        metadata={"help": "Number of collected files"}
    )
    filter_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": "Filter threshold"}
    )
    filter_insufficient: Optional[float] = field(
        default=0.0,
        metadata={"help": "Filter insufficient samples"}
    )


# script_args = ScriptArguments()
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

utils.set_seed(script_args.seed)

def calc_sample_ratio(Gs, ps):
    # Gs: list of gradients
    # ps: list of accept rates
    sample_sizes = []
    for G, p in zip(Gs, ps):
        if G == 0 or p == 0:
            sample_size = 0
        else:
            sample_size = G / (np.sqrt(p + script_args.alpha / np.power(p, script_args.beta - 1.0)))    
        sample_sizes.append(sample_size)
    total = sum(sample_sizes)
    sample_sizes = [sample_size / total for sample_size in sample_sizes]
    return sample_sizes

all_grads = []
accept_rates = []

for index in range(script_args.num_collect_files):
    with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/accept_rates_{index}.json', 'r') as f:
        accept_rate = json.load(f)
    accept_rates.extend(accept_rate)
    with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/grads_{index}.json', 'r') as f:
        grad = json.load(f)
    all_grads.extend(grad)

script_args.stage_2_samples = min(script_args.stage_2_samples, len(all_grads) * script_args.stage_1_samples)

sample_sizes = calc_sample_ratio(all_grads, accept_rates)

if script_args.filter_threshold > 0:
    sample_sizes = [sample_sizes[i] if accept_rates[i] <= script_args.filter_threshold else 0 for i in range(len(sample_sizes))]
if script_args.filter_insufficient > 0:
    sample_sizes = [sample_sizes[i] if (accept_rates[i] * sample_sizes[i] * script_args.stage_2_samples >= script_args.filter_insufficient) else 0 for i in range(len(sample_sizes))]

# normalize sample sizes
total = sum(sample_sizes)
sample_sizes = [sample_size / total for sample_size in sample_sizes]

with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/sample_sizes_ratio_sample{script_args.stage_1_samples}_a{script_args.alpha}_b{script_args.beta}.json', 'w') as f:
    json.dump(sample_sizes, f, indent=4)
# with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/sample_sizes_ratio_{script_args.local_index}.json', 'r') as f:
#     sample_sizes = json.load(f)

def float_to_int_preserve_sum(arr, N):
    # 1. 初步缩放并四舍五入
    scaled_arr = np.array(arr) * N
    int_arr = np.round(scaled_arr).astype(int)
    # print(int_arr)

    # 2. 计算误差
    error = N - np.sum(int_arr)

    # 3. 误差修正：根据四舍五入前的误差最小调整
    if error != 0:
        # 计算原始浮点数和转换后整数的误差
        residuals = scaled_arr - int_arr
        # 按误差绝对值最大调整
        indices = np.argsort(-residuals if error > 0 else residuals)[:abs(error)]
        int_arr[indices] += np.sign(error)  # 调整以匹配总和

    return int_arr.tolist()

sample_sizes = float_to_int_preserve_sum(sample_sizes, script_args.stage_2_samples)

with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/sample_sizes_sample{script_args.stage_1_samples}_a{script_args.alpha}_b{script_args.beta}.json', 'w') as f:
    json.dump(sample_sizes, f, indent=4)

# samples_per_file = len(sample_sizes) // script_args.num_collect_files
# for index in range(script_args.num_collect_files):
#     with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/sample_sizes_{index}.json', 'w') as f:
#         json.dump(sample_sizes[index * samples_per_file: (index + 1) * samples_per_file], f, indent=4)

# print('Sample sizes:', sample_sizes)
print('done!')