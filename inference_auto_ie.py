import argparse
import os

import pandas as pd

from inference import inference
from modules.utils import load_model_and_tokenizer

MODEL_PATHS = {
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "asclepius": "starmpcc/Asclepius-7B",
    "clinical-camel-7b": "augtoma/qCammel-13",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "alpaca-7b": "chavinlo/alpaca-native",
    "medalpaca-7b": "medalpaca/medalpaca-7b",
}

INFORMATION_EXTRACTION_TASKS = {
    11: "medication_extraction",
    15: "concept_problem_extraction",
    12: "concept_test_extraction",
    13: "concept_treatment_extraction",
    16: "risk_factor_cad_extraction",
    14: "drug_extraction",
}

EVAL_MODES = {
    11: "list",
    12: "list",
    13: "list",
    14: "list",
    15: "list",
    16: "list",
}


def inference_all(args):

    model_path = MODEL_PATHS[args.model]
    tasks_ids = list(args.tasks_idxs.keys())
    module, tokenizer, model_config = load_model_and_tokenizer(
        model_path, eval_type=EVAL_MODES[tasks_ids[0]]
    )

    kwargs_list = []

    print("Loading instructions...")
    df = pd.read_csv(args.instruction_csv)
    for i in df.index:
        row = df.loc[i]

        if row.iloc[0] != args.annotator and args.annotator != "all":
            continue

        annotator_name = row.iloc[0]
        annotator_name = annotator_name.replace(" ", "_")

        for j in range(1, len(df.columns)):

            if args.tasks_idxs is not None and j not in args.tasks_idxs.keys():
                continue

            dataset_name = args.tasks_idxs[j]
            output_path = os.path.join(
                args.root_dir, dataset_name, args.model, annotator_name
            )

            instruction = row.iloc[j]

            kwargs = {
                "dataset_name": dataset_name,
                "eval_mode": EVAL_MODES[j],
                "model_path": model_path,
                "root_path": "./datasets",
                "output_path": output_path,
                "instruction": instruction,
                "annotator": annotator_name,
                "truncation_strategy": "split",
                "model": module,
                "tokenizer": tokenizer,
                "model_config": model_config,
            }
            kwargs_list.append(kwargs)

    print(f"Total number of inferences: {len(kwargs_list)}")

    for i, kwargs in enumerate(kwargs_list):
        if os.path.exists(os.path.join(kwargs["output_path"], "predict_logit.json")):
            print(f"Skipping inference {i+1}/{len(kwargs_list)}; already exists!")
            continue

        print(f"Running inference {i+1}/{len(kwargs_list)}")
        try:
            inference(**kwargs)
        except Exception as e:
            print(f"Error in inference {i+1}/{len(kwargs_list)}")
            print(kwargs["dataset_name"])
            print(kwargs["model_path"])
            print(e)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-7b",
        choices=[
            "Llama-2-7b-chat-hf",
            "Llama-2-13b-chat",
            "mistral-7b",
            "asclepius",
            "clinical-camel-7b",
            "alpaca-7b",
            "medalpaca-7b",
        ],
    )
    parser.add_argument("--annotator", type=str, required=True)

    parser.add_argument("--tasks_idxs", type=dict, default=INFORMATION_EXTRACTION_TASKS)

    parser.add_argument("--root_dir", type=str, default="./results/")
    parser.add_argument(
        "--instruction_csv", type=str, default="./instructions/instructions_from_experts.csv"
    )

    args = parser.parse_args()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)

    inference_all(args)


if __name__ == "__main__":
    main()
