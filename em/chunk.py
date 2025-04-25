import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    n: Optional[int] = field(
        default=4,
        metadata={"help": "chunk size"}
    )
    sample_size_file: Optional[str] = field(
        default="sample_size.json",
        metadata={"help": "sample size file"}
    )
    file_path: Optional[str] = field(
        default="",
        metadata={"help": "path to save the sample size"}
    )
    save_file: Optional[str] = field(
        default="sample_size.json",
        metadata={"help": "save file name"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

with open(f'{script_args.file_path}/{script_args.sample_size_file}', 'r') as f:
    sample_sizes = json.load(f)

sample_sizes = [(sample_size + script_args.n - 1) // script_args.n for sample_size in sample_sizes]

with open(f'{script_args.file_path}/{script_args.save_file}', 'w') as f:
    json.dump(sample_sizes, f, indent=4)
