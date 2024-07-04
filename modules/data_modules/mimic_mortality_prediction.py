import os
from datasets import load_from_disk
from ipdb import set_trace
from modules.data_modules.data_module import AbstractDataModule

GT_LABEL_SPACE = {0: "No", 1: "Yes"}

ANNOTATOR_LABEL_SPACES = {
    "default": {0: "No", 1: "Yes"},
    "Annotator_4": {0: "no", 1: "yes"},
    "Annotator_9": {0: "NO", 1: "YES"},
}


class MimicMortalityPrediction(AbstractDataModule):

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
        super().__init__(
            root_path,
            tokenizer,
            truncation_strategy,
            model_config,
            total_samples,
            is_alpaca,
            is_medtuned,
        )
        pred_set_path = os.path.join(root_path, "mimic", "inhospital_mortality")
        full_path = os.path.join(pred_set_path, "test_subsample")
        self.pred_set = load_from_disk(full_path).select_columns(
            ["text", "INHOSP_MORT"]
        )

        self.pred_set = self.pred_set.rename_column("text", "input")
        self.pred_set = self.pred_set.rename_column("INHOSP_MORT", "output")
        # map 0 to 'No' and 1 to 'Yes' in output 
        self.pred_set = self.pred_set.map(
            lambda example: {
                "text": example["input"],
                "output": "No" if example["output"] == 0 else "Yes",
            }
        )
        self.original_label_space = GT_LABEL_SPACE
        self.annotator_label_spaces = ANNOTATOR_LABEL_SPACES

        if self.total_samples is not None and self.total_samples < len(self.pred_set):
            # randomly select total_samples from pred_set
            self.pred_set = self.pred_set.shuffle(seed=42)
            self.pred_set = self.pred_set.select(range(self.total_samples))
            print("Total samples now: ", len(self.pred_set))