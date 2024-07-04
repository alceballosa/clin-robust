import argparse
import json
import os
import pdb
import sys

import torch
import tqdm
from ipdb import set_trace
from lightning import Fabric, Trainer, seed_everything

sys.path.append("./modules")
from modules.utils import (
    do_logit_based_evaluation,
    load_data_module,
    load_model_and_tokenizer,
)

torch.set_float32_matmul_precision("high")


def inference(
    dataset_name: str,
    model_path: str,
    root_path: str,
    output_path: str,
    instruction: str,
    annotator: str,
    truncation_strategy: list,
    seed: int = 42,
    total_samples: int = None,
    model=None,
    tokenizer=None,
    model_config=None,
    eval_mode="logit",  # or "text"
):

    torch.set_float32_matmul_precision("high")
    if seed is not None:
        seed_everything(seed)

    if not os.path.exists(path=output_path):
        os.makedirs(output_path)

    if model is None or tokenizer is None or model_config is None:
        model, tokenizer, model_config = load_model_and_tokenizer(
            model_path, eval_type=eval_mode
        )

    is_alpaca = "alpaca" in model_path
    data_module = load_data_module(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        root_path=root_path,
        truncation_strategy=truncation_strategy,
        model_config=model_config,
        total_samples=total_samples,
        is_alpaca=is_alpaca,
    )

    data_module.customize_instructions(instruction=instruction, annotator=annotator)
    data_module.prepare_data()

    print("Start inference...")
    # LIST BASED EVALUATION FOR INFORMATION EXTRACTION
    if eval_mode == "list":
        fabric = Fabric(accelerator="gpu", devices=1, precision="bf16-mixed")
        dataloader = data_module.predict_dataloader()
        model = fabric.setup(model)
        dataloader = fabric.setup_dataloaders(dataloader)
        model.eval()
        all_outputs, all_idxs, all_inputs, all_gold = [], [], [], []

        for batch in tqdm.tqdm(dataloader, desc="Inference"):

            input_text = tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )
            idxs = batch["idx"]
            idxs, _, pred_ids = model.predict_step(batch)

            preds_text = tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            all_outputs.extend(preds_text)
            all_idxs.extend(idxs.cpu().numpy())
            all_inputs.extend(input_text)
            all_gold.extend(batch["labels"])
        output_dict = {}
        for idx, input_text, pred, gold in zip(
            all_idxs, all_inputs, all_outputs, all_gold
        ):
            idx = int(idx)
            if idx not in output_dict:
                output_dict[idx] = []
            output_dict[idx].append(
                {"inputs": input_text, "pred_list": pred, "gold_list": gold}
            )
        with open(
            os.path.join(output_path, "predict_logit.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(output_dict, f, indent=4)
    # LOGIT-BASED EVALUATION FOR CLF
    elif eval_mode == "logit":

        fabric = Fabric(accelerator="gpu", devices=1, precision="bf16-mixed")
        dataloader = data_module.predict_dataloader()
        model = fabric.setup(model)
        dataloader = fabric.setup_dataloaders(dataloader)
        # set eval mode
        model.eval()
        all_pred_classes, all_gold_classes = [], []
        all_pred_text, all_gold_text = [], []
        all_idxs = []
        all_inputs = []
        all_full_pred_text = []
        all_probabilities = []
        for batch in tqdm.tqdm(dataloader, desc="Inference"):
            pred_classes, gold_classes, preds_text, golds_text, probabilities = (
                do_logit_based_evaluation(batch, model, tokenizer)
            )
            input_text = batch["input_text"]
            # TODO: try tokenizer pad token id
            if "flan" in model_path:
                input_text = [
                    input_text[i]
                    .replace("  ", " ")
                    .replace("</s>", "")
                    .replace("<unk>", " ")
                    for i in range(len(input_text))
                ]
            preds_text = [
                preds_text[i].replace(input_text[i].replace("<s>", ""), "")
                for i in range(len(preds_text))
            ]
            golds_text = [
                golds_text[i].replace(input_text[i].replace("<s>", ""), "")
                for i in range(len(golds_text))
            ]
            input_ids_full_preds = batch["input_ids"]
            attention_mask_full_preds = batch["attention_mask"]
            batch_full_text = {
                "input_ids": input_ids_full_preds,
                "attention_mask": attention_mask_full_preds,
                "idx": batch["idx"],
            }

            idxs, _, pred_ids = model.predict_step(batch_full_text)
            full_preds_text = tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            all_pred_classes.extend(pred_classes)
            all_gold_classes.extend(gold_classes)
            all_pred_text.extend(preds_text)
            all_gold_text.extend(golds_text)
            all_full_pred_text.extend(full_preds_text)
            all_inputs.extend(input_text)
            all_idxs.extend(batch["idx"].cpu().numpy())
            all_probabilities.extend(probabilities)
        output_dict = {}
        for (
            idx,
            input_text,
            pred_class,
            gold_class,
            pred_text,
            gold_text,
            full_pred_text,
            probabilities,
        ) in zip(
            all_idxs,
            all_inputs,
            all_pred_classes,
            all_gold_classes,
            all_pred_text,
            all_gold_text,
            all_full_pred_text,
            all_probabilities,
        ):
            idx = int(idx)
            if idx not in output_dict:
                output_dict[idx] = []
            output_dict[idx].append(
                {
                    "inputs": input_text,
                    "probabilities": probabilities,
                    "pred_class_from_probas": int(pred_class),
                    "gold_class": int(gold_class),
                    # "pred_class_": pred_text,
                    # "gold_text": gold_text,
                    "generated_text": full_pred_text,
                }
            )
        with open(
            os.path.join(output_path, "predict_logit.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(output_dict, f, indent=4)
    else:
        raise ValueError("Invalid eval_mode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="mimic_mortality_prediction"
    )
    parser.add_argument(
        "--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_samples", type=int, default=None)

    parser.add_argument("--root_path", type=str, default="./datasets")
    parser.add_argument("--output_path", type=str, default="./results/test")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Whether or not the patient will die during the course of a stay in the hospital given notes from the first 48 hours?",
    )
    parser.add_argument(
        "--truncation_strategy",
        type=str,
        default="split",
        choices=["filter", "truncate", "split"],
    )

    args = parser.parse_args()

    inference(**vars(args))


if __name__ == "__main__":
    main()
