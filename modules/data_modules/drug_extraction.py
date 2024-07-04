import os

from datasets import load_from_disk

from modules.data_modules.data_module import AbstractDataModule


def get_targets(row):
    concepts = row["concepts"]
    targets = []
    for concept in concepts:
        if concept["category"] == "Drug":
            targets.append(concept["text"])
    return targets

class DrugExtraction(AbstractDataModule):

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

        # drugs for synonyms come from # TODO: 
        pred_set_path = os.path.join(root_path, "n2c2/adverse-drug-effects-2018")
        self.pred_set = load_from_disk(pred_set_path)["test"].select_columns(
            ["text", "concepts"]
        )
        targets = [get_targets(row) for row in self.pred_set]
        self.pred_set = self.pred_set.rename_column("text", "input")
        self.pred_set = self.pred_set.remove_columns("concepts")
        self.pred_set = self.pred_set.add_column("output", targets)
        self.is_list_output = True
        if self.total_samples is not None and self.total_samples < len(self.pred_set):
            # randomly select total_samples from pred_set
            self.pred_set = self.pred_set.shuffle(seed=42)
            self.pred_set = self.pred_set.select(range(self.total_samples))
            print("Total samples now: ", len(self.pred_set))


