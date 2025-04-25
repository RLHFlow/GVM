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

# os.environ['CUDA_VISIBLE_DEVICES'] = '8'

@dataclass
class ScriptArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={"help": "Max length of newly generated tokens"}
    )
    model_name_or_path: Optional[str] = field(
        default='Qwen/Qwen2.5-Math-7B',
        metadata={"help": "Model name or path"}
    )
    epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of epochs"}
    )
    alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "Penalty weight alpha"}
    )
    beta: Optional[float] = field(
        default=2.0,
        metadata={"help": "Penalty weight beta"}
    )
    lr: Optional[float] = field(
        default=0.5,
        metadata={"help": "Learning rate"}
    )
    start: Optional[int] = field(
        default=0,
        metadata={"help": "Start index"}
    )
    end: Optional[int] = field(
        default=100000,
        metadata={"help": "End index"}
    )
    stage_1_samples: Optional[int] = field(
        default=8,
        metadata={"help": "Number of samples for stage 1 per example"}
    )
    stage_2_samples_max: Optional[int] = field(
        default=24,
        metadata={"help": "Max number of samples for stage 2 per example"}
    )
    local_index: Optional[int] = field(
        default=1,
        metadata={"help": "the local index of the agent"}
    )
    iter: Optional[int] = field(
        default=2,
        metadata={"help": "the iteration of the experiment"}
    )
    correct_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": "the threshold for correctness"}
    )
    model_prefix: Optional[str] = field(
        default='Qwen7B',
        metadata={"help": "the model prefix"}
    )
    suffix: Optional[str] = field(
        default='',
        metadata={"help": "the suffix"}
    )
    system_prompt: Optional[str] = field(
        default='qwen25-math-cot',
        metadata={"help": "the system prompt type"}
    )

# script_args = ScriptArguments()
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

utils.set_seed(script_args.seed)

# ds = load_dataset('HuggingFaceH4/MATH-500')['test']
with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/stage_1_collected_data_{script_args.local_index}.json', 'r') as f:
    ds = json.load(f)

ds = Dataset.from_list(ds)

with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/sample_sizes_{script_args.local_index}.json', 'r') as f:
    sample_sizes = json.load(f)

script_args.end = min(script_args.end, len(ds))
sample_sizes = sample_sizes[script_args.start:script_args.end]

max_sample_size = max(sample_sizes)
max_sample_size = min(max_sample_size, script_args.stage_2_samples_max)
print('Max sample size:', max_sample_size)
for i in range(len(sample_sizes)):
    sample_sizes[i] = min(sample_sizes[i], max_sample_size)

script_args.end = min(len(ds), script_args.end)
ds = ds.select(range(script_args.start, script_args.end))
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
llm = LLM(script_args.model_name_or_path, 
        #   gpu_memory_utilization=0.4,
          dtype=torch.bfloat16)

def stage_2_sampling_flat(sample_sizes):
    sampling_params = SamplingParams(
        max_tokens=script_args.max_length,
        temperature=1.0,
        n=1,
    )
    prompts = []
    for i, item in enumerate(ds):
        if script_args.system_prompt:
            conv = [
                {'role': 'system', 'content': utils.SYSTEM_PROMPTS[script_args.system_prompt]},
                {'role': 'user', 'content': item['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}
            ]
        else:
            conv = [{'role': 'user', 'content': item['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for _ in range(sample_sizes[i]-script_args.stage_1_samples):
            prompts.append(conv_chat)
    outputs = llm.generate(prompts, sampling_params)

    idx_sum = 0
    new_outputs = []
    for i in range(len(sample_sizes)):
        new_outputs.append([])
        for idx in range(idx_sum, idx_sum + (sample_sizes[i]-script_args.stage_1_samples)):
            new_outputs[-1].append(outputs[idx].outputs[0].text)

        idx_sum += max(sample_sizes[i]-script_args.stage_1_samples, 0)

    return new_outputs

def stage_2_sampling_max(sample_sizes):
    if script_args.stage_1_samples >= max_sample_size:
        return [[] for _ in range(len(sample_sizes))], [[] for _ in range(len(sample_sizes))]
    
    new_outputs = []
    prompts = []

    sampling_params = SamplingParams(
        max_tokens=script_args.max_length,
        temperature=1.0,
        # n=max_sample_size,
        n=max_sample_size-script_args.stage_1_samples,
        # n=8,
    )

    for i in range(len(sample_sizes)):
        if script_args.system_prompt:
            conv = [
                {'role': 'system', 'content': utils.SYSTEM_PROMPTS[script_args.system_prompt]},
                {'role': 'user', 'content': ds[i]['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}
            ]
        else:
            conv = [{'role': 'user', 'content': ds[i]['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompts.append(conv_chat)
    
    outputs = llm.generate(prompts, sampling_params)
    for i in range(len(sample_sizes)):
        new_outputs.append([outputs[i].outputs[j].text for j in range(sample_sizes[i]-script_args.stage_1_samples)])
    all_outputs = [[output.text for output in outputs[i].outputs] for i in range(len(outputs))]

    return new_outputs, all_outputs

def stage_2_sampling(sample_sizes):
    new_outputs = []

    for i in range(len(sample_sizes)):
        if sample_sizes[i] == 0:
            new_outputs.append([])
            continue
        sampling_params = SamplingParams(
            max_tokens=script_args.max_length,
            temperature=1.0,
            n=sample_sizes[i],
            # n=8,
        )
        if script_args.system_prompt:
            conv = [
                {'role': 'system', 'content': utils.SYSTEM_PROMPTS[script_args.system_prompt]},
                {'role': 'user', 'content': ds[i]['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}
            ]
        else:
            conv = [{'role': 'user', 'content': ds[i]['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompts = [conv_chat]
        outputs = llm.generate(prompts, sampling_params)
        new_outputs.append([output.text for output in outputs[0].outputs])

    return new_outputs

# use flat sampling
stage_2_outputs = stage_2_sampling_flat(sample_sizes)

#
# use max sampling
# stage_2_outputs, all_outputs = stage_2_sampling_max(sample_sizes)

# with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/stage_2_allOutputs_{script_args.local_index}.json', 'w', encoding='utf-8') as f:
#     json.dump(all_outputs, f, indent=4, ensure_ascii=False)
#

# with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/stage_2_allOutputs_{script_args.local_index}.json', 'r', encoding='utf-8') as f:
#     all_outputs = json.load(f)
# stage_2_outputs = []
# for i in range(len(sample_sizes)):
#     stage_2_outputs.append([all_outputs[i][j] for j in range(sample_sizes[i]-script_args.stage_1_samples)])

with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/stage_1_collected_data_all_{script_args.local_index}.json', 'r') as f:
    stage_1_collected_data_all = json.load(f)

for i in range(len(stage_2_outputs)):
    stage_2_outputs[i].extend(stage_1_collected_data_all[i]['outputs'])
    # all_outputs[i].extend(stage_1_collected_data_all[i]['outputs'])

stage_2_collected_data = []
corrects_2 = []
total_samples = 0
for i, item in enumerate(tqdm(ds, desc='Collecting stage 2 samples')):
    collected_data = {
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': []
    }
    problem_corrects = []
    for j in range(sample_sizes[i]):
        # correct = reward_labeling.is_equal(outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        # correct = reward_labeling.is_equal(stage_2_outputs[i][j], item['answer'], dataset_name='math500')
        try:
            correct = utils.check_correct(stage_2_outputs[i][j], item['answer'], i, threshold=script_args.correct_threshold)
        except utils.TimeoutError as e:
            correct = False
        if correct:
            problem_corrects.append(j)
            # collected_data['outputs'].append(outputs[i].outputs[j].text)
            collected_data['outputs'].append(stage_2_outputs[i][j])
    corrects_2.append(problem_corrects)
    stage_2_collected_data.append(collected_data)
    total_samples += len(collected_data['outputs'])

print('Total collected samples:', total_samples)
# stage_2_collected_data_ds = Dataset.from_list(stage_2_collected_data)
# stage_2_collected_data_ds.save_to_disk('data/stage_2_collected_data')
with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/stage_2_collected_data_{script_args.local_index}.json', 'w', encoding='utf-8') as f:
    json.dump(stage_2_collected_data, f, indent=4, ensure_ascii=False)

print('done!')