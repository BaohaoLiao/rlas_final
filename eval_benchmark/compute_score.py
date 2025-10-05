from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser

"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    record_path: Optional[str] = field(
        default="record.txt",
        metadata={"help": "the location of the output file"},
    )

    
def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

import numpy as np

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ds = load_dataset("json", data_files=script_args.dataset_path, split="train")

all_scores = []
for i in range(len(ds)):
    tmp_scores = []
    all_responses = ds[i]["responses"]
    ground_truth = ds[i]["gt"]
    for response in all_responses:
        score = compute_score(response, ground_truth)
        tmp_scores.append(score)
    all_scores.append(tmp_scores)

with open(script_args.record_path, "w") as f:
    rounded_scores = np.round(np.mean(all_scores), 4)
    f.write(script_args.dataset_path + " " + str(rounded_scores) + "\n")
