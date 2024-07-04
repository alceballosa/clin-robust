import os

from datasets import load_from_disk

from modules.data_modules.data_module import AbstractDataModule

GT_LABEL_SPACE = {0: "not met", 1: "met"}

ANNOTATOR_LABEL_SPACES = {
    "default": {0: "No", 1: "Yes"},
}


class CohortAbdominalClassification(AbstractDataModule):

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

        pred_set_path = os.path.join(root_path, "n2c2", "cohort-selection-2018")
        self.pred_set = load_from_disk(pred_set_path)["test"].select_columns(
            ["text", "ABDOMINAL"]
        )

        self.pred_set = self.pred_set.rename_column("text", "input")
        self.pred_set = self.pred_set.rename_column("ABDOMINAL", "output")
        self.original_label_space = GT_LABEL_SPACE
        self.annotator_label_spaces = ANNOTATOR_LABEL_SPACES

        if self.total_samples is not None and self.total_samples > len(self.pred_set):
            # randomly select total_samples from pred_set
            self.pred_set = self.pred_set.shuffle(seed=42)
            self.pred_set = self.pred_set.select(range(self.total_samples))
            print("Total samples now: ", len(self.pred_set))
