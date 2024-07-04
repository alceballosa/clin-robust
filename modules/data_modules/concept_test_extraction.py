import os

from datasets import load_from_disk

from modules.data_modules.data_module import AbstractDataModule


class ConceptTestExtraction(AbstractDataModule):

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

        # drugs for synonyms come from https://docs.google.com/spreadsheets/d/1nFZKPhsiXxWN5jidjjwVBd5l4T_kdTrO/edit#gid=1687788618

        pred_set_path = os.path.join(root_path, "n2c2/relation-challenge-2010")
        self.pred_set = load_from_disk(pred_set_path)["test"].select_columns(
            ["text", "concepts"]
        )

        self.pred_set = self.pred_set.rename_column("text", "input")
        # get processed targets
        all_targets = [get_spans_list(targets) for targets in self.pred_set["concepts"]]
        self.pred_set = self.pred_set.remove_columns("concepts")
        self.pred_set = self.pred_set.add_column("output", all_targets)
        self.is_list_output = True
        if self.total_samples is not None and self.total_samples < len(self.pred_set):
            # randomly select total_samples from pred_set
            self.pred_set = self.pred_set.shuffle(seed=42)
            self.pred_set = self.pred_set.select(range(self.total_samples))
            print("Total samples now: ", len(self.pred_set))


def get_spans_list(concepts):
    """
    Puts the target risk factors in a single list.
    """
    spans = []
    for concept in concepts:
        if concept["t"][0] == "test":
            spans.append(concept["c"][0])
    return spans
