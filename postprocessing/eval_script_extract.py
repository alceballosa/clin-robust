import json
import os

import pandas as pd
import tqdm
from task_parser_mapping import get_parser

DATASET_MAP = {
    "drug_extraction": "Drug Extraction",
    "medication_extraction": "Medication Extraction",
    "concept_treatment_extraction": "Concept Treatment Extraction",
    "concept_problem_extraction": "Concept Problem Extraction",
    "concept_test_extraction": "Concept Test Extraction",
    "risk_factor_cad_extraction": "Risk Factor CAD Extraction",
}

tasks = [
    "drug_extraction",
    "medication_extraction",
    "concept_treatment_extraction",
    "concept_problem_extraction",
    "concept_test_extraction",
    "risk_factor_cad_extraction",
]

models = [
    "Llama-2-7b-chat-hf",
    "clinical-camel-7b",
    "mistral-7b",
    "asclepius",
    "medalpaca-7b",
    "alpaca-7b",
    "Llama-2-13b-chat",
]

ANNOTATORS = [
    "Annotator_1",
    "Annotator_2",
    "Annotator_3",
    "Annotator_4",
    "Annotator_5",
    "Annotator_6",
    "Annotator_7",
    "Annotator_8",
    "Annotator_9",
    "Annotator_10",
    "Annotator_11",
    "Annotator_12",
]

EVAL_DIR = "./results/"
EVAL_DIR = "/work/frink/private_datasets/bionlp_results_extraction/"


def get_scores():
    base_results_dir = EVAL_DIR
    # only look at drug dataset for now.
    results = []
    for dataset_name in tasks:
        for model in models:
            print(model)
            model_dataset_path = os.path.join(
                base_results_dir, dataset_name.replace(" ", "_").lower(), model
            )
            if not os.path.exists(model_dataset_path):
                print(f"Skipping {model} for {dataset_name}, not run yet")
                continue
            annotators = os.listdir(model_dataset_path)
            for annotator in tqdm.tqdm(annotators, total=len(annotators)):
                parser = get_parser(
                    DATASET_MAP[dataset_name], model=model, annotator=annotator
                )
                output_path = os.path.join(model_dataset_path, annotator)
                if not os.path.exists(os.path.join(output_path, "predict_logit.json")):
                    print(f"Skipping {output_path}, not run yet")
                    continue

                else:
                    with open(os.path.join(output_path, "predict_logit.json")) as f:
                        outputs = json.load(f)

                for k, v in outputs.items():
                    gt_correctness, pred_correctness = parser.get_sample_stats(v)
                    result_processed = {
                        "dataset": dataset_name,
                        "model": model,
                        "annotator": annotator,
                        "instance": k,
                        "gt_correctness": gt_correctness,
                        "pred_correctness": pred_correctness,
                    }
                    results.append(result_processed)

    results_df = pd.DataFrame(results)
    results_df.to_csv("./postprocessing/processed_csvs/extraction.csv", index=False)
    return results_df


if __name__ == "__main__":
    get_scores()
