import json
import os
import traceback

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score

tasks = {
    "classification": [
        "cohort_drug_abuse_classification",
        "cohort_alcohol_abuse_classification",
        "cohort_english_classification",
        "cohort_make_decisions_classification",
        "cohort_abdominal_classification",
        "obesity_classification",
        "diabetes_mellitus_classification",
        "asthma_classification",
        "cad_classification",
        "mimic_mortality_prediction",
    ]
}

models = [
    "asclepius",
    "clinical-camel-7b",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat",
    "mistral-7b",
    "alpaca-7b",
    "medalpaca-7b",
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

def chunking_output(result_dict):
    preds = result_dict["pred_class"]
    prob = result_dict["probabilities"]
    # Check if all preds are 0

    if len(preds) > 1:
        n = len(preds)
        max_prob_0 = max(p[0] for p in prob)
        max_prob_1 = max(p[1] for p in prob)
        mean_prob_0 = np.mean([p[0] for p in prob])
        mean_prob_1 = np.mean([p[1] for p in prob])

        avg_prob_0 = (max_prob_0 + mean_prob_0 * (n / 2)) / (1 + (n / 2))
        avg_prob_1 = (max_prob_1 + mean_prob_1 * (n / 2)) / (1 + (n / 2))

        return [avg_prob_0, avg_prob_1]
    else:
        return prob[0]


def get_scores():
    base_results_dir = EVAL_DIR
    # only look at drug dataset for now.
    results = []
    for task_type, datasets in tasks.items():

        for dataset_name in datasets:
            print(dataset_name)
            for model in models:
                print(model)
                model_dataset_path = os.path.join(
                    base_results_dir, dataset_name.replace(" ", "_").lower(), model
                )
                if not os.path.exists(model_dataset_path):
                    continue
                annotators = os.listdir(model_dataset_path)
                for annotator in tqdm.tqdm(annotators, total=len(annotators)):
                    annotator_scores = []
                    output_path = os.path.join(model_dataset_path, annotator)
                    if not os.path.exists(
                        os.path.join(output_path, "predict_logit.json")
                    ):
                        print(f"Skipping {output_path}, not run yet")
                        continue
                    else:
                        with open(os.path.join(output_path, "predict_logit.json")) as f:
                            try:
                                outputs = json.load(f)
                            except json.decoder.JSONDecodeError:
                                set_trace()

                        for k, v in outputs.items():
                            result_for_sample = {
                                "task_type": task_type,
                                "dataset": dataset_name,
                                "model": model,
                                "annotator": annotator,
                                "instance": k,
                                "inputs": [i["inputs"] for i in v],
                                "gold_class": v[0]["gold_class"],
                                "pred_class": [i["pred_class_from_probas"] for i in v],
                                "probabilities": [i["probabilities"][0] for i in v],
                            }
                            # aggregate the chunks
                            avg_probs = chunking_output(result_for_sample)
                            # print(avg_probs)

                            result_processed = {
                                "id": k,
                                "task_type": task_type,
                                "dataset": dataset_name,
                                "model": model,
                                "annotator": annotator,
                                "instance": k,
                                "gold_class": v[0]["gold_class"],
                                # "correct": correct,
                                # "pred": prediction,
                                "probabilities": avg_probs,
                            }
                            annotator_scores.append(result_processed)
                            results.append(result_processed)

    results = pd.DataFrame(results)
    results.to_csv(f"./postprocessing/processed_csvs/classification.csv", index=False)
    return results


if __name__ == "__main__":
    get_scores()
