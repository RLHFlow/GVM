from datasets import Dataset, load_from_disk
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, AutoTokenizer
import utils
import os

@dataclass
class ScriptArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    train_size: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of training samples"}
    )
    num_collect_files: Optional[int] = field(
        default=8,
        metadata={"help": "Number of collected files"}
    )
    iter: Optional[int] = field(
        default=1,
        metadata={"help": "the iteration of the experiment"}
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

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/sample_sizes.json') as f:
    sample_sizes = json.load(f)

# all_data = []
# for index in range(script_args.num_collect_files):
#     with open(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/stage_2_collected_data_{index}.json', 'r') as f:
#         data = json.load(f)
#     all_data.extend(data)

# print('Total number of problems:', len(all_data))
all_data = load_from_disk(f'data/{script_args.model_prefix}/{script_args.suffix}/train_ds')
all_data = all_data.select(range(len(sample_sizes)))

new_data = []
for i, item in enumerate(all_data):
    # for output in item['outputs']:
    new_item = {
        "problem": item['problem'],
        "answer": item['answer']
    }
    for _ in range(sample_sizes[i]):
        new_data.append(new_item)

print('Total number of samples:', len(new_data))

ds = Dataset.from_list(new_data)

instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')

def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop('problem')

        question = question + ' ' + instruction_following

        # We set the data_source as MATH so that we can use the reward model designed for MATH dataset
        
        # reward_model =  example['reward_model']
        reward_model = {
            "style": "rule",
            "ground_truth": example['answer']
        }

        data = {
            "data_source": 'numina_math',
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "ability": "math",
            "reward_model": reward_model,
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data

    return process_fn

def able_to_extract(example):
    if len(tokenizer.encode(example['problem'])) > 700:
        return False
    # if last_boxed_only_string(example["response"]):
    #     return True
    return True

ds.filter(able_to_extract)
ds = ds.map(function=make_map_fn('train'), with_indices=True)

# remove_columns = ds.column_names
# ds = ds.map(
#     lambda x: {
#         "conversations": [
#             {'role': 'system', 'content': utils.SYSTEM_PROMPTS[script_args.system_prompt]},
#             {'role': 'user', 'content': x['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'},
#             {'role': 'assistant', 'content': x['output']}
#         ]
#     },
#     remove_columns=remove_columns
# )

script_args.train_size = min(script_args.train_size, len(ds))
ds = ds.shuffle(seed=script_args.seed).select(range(script_args.train_size))
# ds.save_to_disk(f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}/train_data')
local_dir = f'data/{script_args.model_prefix}/{script_args.suffix}/data_{script_args.iter}'
ds.to_parquet(os.path.join(local_dir, 'train.parquet'))