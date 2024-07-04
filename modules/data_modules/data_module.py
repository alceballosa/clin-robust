import json
import math
import os
import pdb
from ipdb import set_trace
import lightning as L
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from lightning.pytorch.utilities.data import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS


class AbstractDataModule(L.LightningDataModule):

    def __init__(
        self,
        root_path,
        tokenizer,
        truncation_strategy,
        model_config,
        total_samples=None,
        is_alpaca=False,
        is_medtuned=False,
    ):
        super(AbstractDataModule).__init__()

        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.tokenizer = tokenizer

        self.truncation_strategy = truncation_strategy
        self.batch_size = model_config["batch_size"]
        self.max_length = model_config["max_length"]
        self.max_new_tokens = model_config["max_new_tokens"]
        self.prompt_format = model_config["prompt_format"]

        self.instruction: str
        self.annotator: str
        self.pred_set = None
        self.annotator_label_spaces: dict = None
        self.annotator_label_space: dict
        self.total_samples = total_samples
        self.is_alpaca: bool = is_alpaca
        self.is_medtuned: bool = is_medtuned
        self.is_list_output: bool = False

    def augment_instruction(self, text):
        formatted_prompt = self.prompt_format.format(
            text=text, instruction=self.instruction
        )
        if self.is_list_output and not self.is_medtuned:
            formatted_prompt = formatted_prompt.strip() + "\nList of outputs: ['"
        return formatted_prompt

    def prepare_data(self):

        assert self.pred_set is not None
        if self.instruction is None:
            raise ValueError("Instruction is not set")

        idxs = list(range(len(self.pred_set)))
        self.pred_set = self.pred_set.add_column(name="idx", column=idxs)

        if self.truncation_strategy == "filter":
            self.filter_all()
        elif self.truncation_strategy == "split":
            splitted_sets = []
            removed_idxs = []
            for i, _ in enumerate(self.pred_set):
                if not self.not_exceed_length(self.pred_set[i]):

                    tokenized_example = self.tokenizer(
                        text=self.pred_set[i]["input"], truncation=False
                    )["input_ids"]

                    example_length = len(tokenized_example)
                    total_length = len(
                        self.tokenizer(
                            text=self.augment_instruction(self.pred_set[i]["input"]),
                            truncation=False,
                        )["input_ids"]
                    )
                    prompt_length = total_length - example_length
                    max_length_per_split = (
                        self.max_length - self.max_new_tokens - prompt_length
                    )

                    num_splits = math.ceil(example_length / max_length_per_split)
                    split_length = math.ceil(example_length / num_splits)

                    splitted_examples = []
                    for j in range(num_splits):
                        start = j * split_length
                        end = min((j + 1) * split_length, example_length)
                        splitted_examples.append(
                            self.tokenizer.decode(tokenized_example[start:end])
                        )

                    for j in range(0, num_splits):
                        new_example = self.pred_set[i].copy()
                        new_example["input"] = splitted_examples[j]
                        splitted_sets.append(new_example)

                    removed_idxs.append(i)

            selected_idxs = [idx for idx in idxs if idx not in removed_idxs]
            self.pred_set = self.pred_set.select(selected_idxs)

            splitted_sets = Dataset.from_list(splitted_sets)
            print(f"new {len(splitted_sets)} Splitted examples")
            self.pred_set = concatenate_datasets([self.pred_set, splitted_sets])
        elif self.truncation_strategy == "truncate":
            pass
        else:
            raise ValueError(f"Invalid truncation strategy: {self.truncation_strategy}")

    def load_instruction(self, instruction: str):
        self.instruction = instruction

    def load_annotator(self, annotator):
        self.annotator = annotator
        # check if annotator_label_spaces is set
        if self.annotator_label_spaces is None:
            return
        if self.annotator in self.annotator_label_spaces:
            self.annotator_label_space = self.annotator_label_spaces[annotator]
        else:
            print(f"Using default label space for {self.annotator}")
            self.annotator_label_space = self.annotator_label_spaces["default"]

    def customize_instructions(self, annotator, instruction):
        self.load_annotator(annotator)
        self.load_instruction(instruction)
        if self.annotator_label_spaces is None:
            return
        reversed_gt_label_space = {v: k for k, v in self.original_label_space.items()}
        all_gts = self.pred_set["output"]
        all_gt_ids = [reversed_gt_label_space[gt] for gt in all_gts]
        all_gts = [self.annotator_label_space[gt] for gt in all_gt_ids]
        self.pred_set = self.pred_set.remove_columns("output")
        self.pred_set = self.pred_set.add_column("output", all_gts)
        self.pred_set = self.pred_set.add_column(
            "label_space", [self.annotator_label_space.values()] * len(self.pred_set)
        )

    def not_exceed_length(self, example):
        text = self.augment_instruction(example["input"])
        return len(self.tokenizer(text=text)["input_ids"]) < (
            self.max_length - self.max_new_tokens
        )

    def filter_all(self):
        length_before = len(self.pred_set)
        self.pred_set = self.pred_set.filter(self.not_exceed_length)
        length_after = len(self.pred_set)
        print(f"Filtered {length_before - length_after} examples")

    def collate_fn(self, batch):
        input_text = [item["input"] for item in batch]
        idxs = [item["idx"] for item in batch]
        input_text = [self.augment_instruction(text) for text in input_text]
        # collated_batch = self.tokenizer(
        #     text=input_text,
        #     padding="longest",
        #     truncation=True,
        #     return_tensors="pt",
        #     max_length=self.max_length - self.max_new_tokens,
        # )
        # collated_batch["idx"] = torch.tensor(idxs)
        input_tokenized = self.tokenizer(
            text=input_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length - self.max_new_tokens,
        )
        # input tokenized to dict:
        collated_batch = {**input_tokenized}
        collated_batch["idx"] = torch.tensor(idxs)
        # add label space data
        if "label_space" in batch[0]:
            input_ids_full_preds = collated_batch["input_ids"]
            attention_mask_full_preds = collated_batch["attention_mask"]
            label_spaces = [item["label_space"] for item in batch]
            output_text = [item["output"] for item in batch]
            labels_cls = torch.ShortTensor(
                [
                    label_space.index(y)
                    for label_space, y in zip(label_spaces, output_text)
                ]
            )

            collated_batch = {}
            collated_batch["idx"] = torch.tensor(idxs)
            collated_batch["label_cls"] = labels_cls
            collated_batch["label_space"] = label_spaces
            collated_batch["input_ids"] = input_ids_full_preds
            collated_batch["attention_mask"] = attention_mask_full_preds
            collated_batch["input_text"] = input_text
        elif "label_space" in batch[0]:
            input_ids_full_preds = collated_batch["input_ids"]
            attention_mask_full_preds = collated_batch["attention_mask"]
            label_spaces = [item["label_space"] for item in batch]
            output_text = [item["output"] for item in batch]
            labels_cls = torch.ShortTensor(
                [
                    label_space.index(y)
                    for label_space, y in zip(label_spaces, output_text)
                ]
            )
            # create one version of the batch for each potential ending (from label spaces)
            all_inputs_per_label = []
            all_masks_per_label = []
            for label in label_spaces[0]:  # all label spaces are the same
                inputs_for_label = []
                for item in input_text:
                    inputs_for_label.append(item + label)
                inputs_for_label = self.tokenizer(
                    text=inputs_for_label,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                    - self.max_new_tokens
                    + 20,  # ensure label fits
                )
                # each entry in the list is a version of the batch
                # concatenated with each of the labels (e.g., yes/no)
                all_inputs_per_label.append(inputs_for_label["input_ids"])
                all_masks_per_label.append(inputs_for_label["attention_mask"])
            collated_batch = {}
            collated_batch["idx"] = torch.tensor(idxs)
            collated_batch["label_cls"] = labels_cls
            collated_batch["label_space"] = label_spaces
            collated_batch["input_ids"] = torch.stack(
                all_inputs_per_label, dim=0
            )
            collated_batch["input_text"] = input_text

            collated_batch["attention_mask"] = torch.stack(
                all_masks_per_label, dim=0
            )
            collated_batch["input_ids_full_preds"] = input_ids_full_preds
            collated_batch["attention_mask_full_preds"] = attention_mask_full_preds

        elif "label_space" in batch[0]:
            label_spaces = [item["label_space"] for item in batch]
            output_text = [item["output"] for item in batch]
            labels_cls = torch.ShortTensor(
                [
                    label_space.index(y)
                    for label_space, y in zip(label_spaces, output_text)
                ]
            )
            label_spaces_ids = [
                self.tokenizer(label_space, padding=False, return_length=True)
                for label_space in label_spaces
            ]

            sample_to = [label_space["length"] for label_space in label_spaces_ids]
            max_seq_len = max([max(leng) for leng in sample_to])
            label_spaces_ids = [
                self.tokenizer(
                    label_space,
                    padding="max_length",
                    max_length=max_seq_len,
                    return_tensors="pt",
                )["input_ids"]
                for label_space in label_spaces
            ]
            output_ids = [
                self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_seq_len,
                    return_tensors="pt",
                )["input_ids"]
                for text in output_text
            ]
            output_ids = torch.stack(output_ids, dim=0)
            label_spaces_ids = torch.stack(label_spaces_ids, dim=0)

            sample_to = torch.ShortTensor([min(lengths) for lengths in sample_to])
            if self.is_alpaca:
                label_spaces_ids = label_spaces_ids[:, :, 1:]  # remove the <s> token
                sample_to -= 1
                max_seq_len -= 1

            # put label info in batch
            
            collated_batch["label_cls"] = labels_cls
            collated_batch["label_spaces_ids"] = label_spaces_ids
            collated_batch["sample_to"] = sample_to
            collated_batch["labels"] = output_ids[:, 0, :]
        else:
            output_text = [item["output"] for item in batch]
            collated_batch["labels"] = output_text
        return collated_batch

    def save_groundtruth(self, output_path):
        assert self.pred_set is not None
        json.dump(
            self.pred_set["output"],
            open(os.path.join(output_path, "groundtruth.json"), "w"),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.pred_set, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
