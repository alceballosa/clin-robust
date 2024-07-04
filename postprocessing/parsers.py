import re
from typing import List, Optional, Tuple, Union

import numpy as np


class BaseParser:
    def parse_output(self, output: str) -> Union[str, int]:
        raise NotImplementedError

    def combine_results(self, results: List[Union[str, int]]) -> Union[str, int]:
        raise NotImplementedError

    def __call__(
        self, outputs: List[str]
    ) -> Tuple[List[Union[str, int]], Union[str, int]]:
        results = [self.parse_output(output) for output in outputs]
        final_result = self.combine_results(results)
        return results, final_result


class ClassificationBinaryParser(BaseParser):
    def __init__(
        self,
        class_info: List[Tuple[Optional[str], Union[str, int]]],
        class_priority: List[Union[str, int]],
    ):
        """
        class_info: a list of tuples of the form
                (<regular expression>, <label>).
            The regular expressions will be checked in order and when a
            match is reached, the corresponding label will be predicted.
            A regular expression of None should be given for the last
            class to catch all previously uncaught outputs. all other
            regular expressions should be strings.

        class_priority: the labels listed in the order of priority: the
            highest priority labels (at the top of the list) will
            superseed lower priority ones during aggregation.
        """
        self.class_info = class_info
        assert len(self.class_info) > 1
        assert self.class_info[-1][0] is None
        self.class_priority = class_priority
        self.class_priority_map = {label: i for i, label in enumerate(class_priority)}

    def parse_output(self, output: str) -> Union[str, int]:
        for regex, label in self.class_info[:-1]:
            assert isinstance(regex, str)
            if re.fullmatch(regex, output, flags=re.DOTALL) is not None:
                return label
        return self.class_info[-1][1]

    def combine_results(self, results: List[Union[str, int]]) -> Union[str, int]:
        priority_indices = [self.class_priority_map[result] for result in results]
        priority_idx = min(priority_indices)
        return self.class_priority[priority_idx]


class ClassificationMultiParser(BaseParser):
    def __init__(
        self,
        class_info: List[Tuple[Optional[str], Union[str, int]]],
        class_priority: List[Union[str, int]],
    ):
        """
        class_info: a list of tuples of the form
                (<regular expression>, <label>).
            The regular expressions will be checked in order and when a
            match is reached, the corresponding label will be predicted.
            A regular expression of None should be given for the last
            class to catch all previously uncaught outputs. all other
            regular expressions should be strings.

        class_priority: the labels listed in the order of priority: the
            highest priority labels (at the top of the list) will
            superseed lower priority ones during aggregation.
        """
        self.class_info = class_info
        # assert len(self.class_info) > 1
        # assert self.class_info[-1][0] is None
        self.class_priority = class_priority
        self.class_priority_map = {label: i for i, label in enumerate(class_priority)}

    def parse_output(self, output: str) -> Union[str, int]:
        for regex, label in self.class_info[:-1]:
            assert isinstance(regex, str)
            if re.fullmatch(regex, output, flags=re.DOTALL) is not None:
                return label
        return self.class_info[-1][1]

    def combine_results(self, results: List[Union[str, int]]) -> Union[str, int]:
        priority_indices = [self.class_priority_map[result] for result in results]
        priority_idx = min(priority_indices)
        return self.class_priority[priority_idx]


class ExtractionParser:
    def __init__(self, mode="strict"):
        """
        Class to do parsing of extractions and get true positives, false positives and false negatives.
        """

    def preprocessing_gt(self, x: str) -> str:
        return x.lower().strip()

    def preprocessing_pred_for_recall(self, x: str) -> str:
        x = x.lower().strip()
        # replace all consecutive white-spaces
        x = re.sub(r"\s+", " ", x)
        # replace all consecutive \n
        x = re.sub(r"\n+", " ", x)
        # replace all ' and "
        x = x.replace("'", ",")
        x = x.replace('"', ",")
        # replace all consecutive ,
        x = re.sub(r",+", ",", x)
        return x

    def parse_ground_truths_in_prediction(
        self, gts: list[str], pred: str, mode: str = "strict"
    ) -> list[bool]:
        """
        Determines whether the extracted predictions include each of the ground truths by
        assessing whether the string for each gt is contained in the prediction string.

        mode = "strict" means every token from the gt instance should be in the prediction to
                be considered as a detection.

        Returns: list of boolean values indicating whether each gt was found in the prediction.
        """
        gts_found = []
        gts = [self.preprocessing_gt(gt) for gt in gts]
        pred = self.preprocessing_gt(pred)
        if mode == "strict":
            for gt in gts:
                if gt in pred:
                    gts_found.append(True)
                else:
                    gts_found.append(False)
        else:
            raise NotImplementedError
        return gts_found

    def preprocessing_pred_for_precision(self, x: str) -> list[str]:
        """
        Given a string output by a model, this function converts it to a list of sub-strings
        in lower-case.

        Sub-strings are tokenized based on the presence of separation tokens such as:

        - ,
        - and
        - or
        - '
        - /
        - "
        - [
        - ]
        """
        x = x.lower().strip()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"\n+", ",", x)
        x = x.replace("'", ",")
        x = x.replace('"', ",")
        x = x.replace("/", ",")
        x = x.replace("and ", ",")
        x = x.replace("or ", ",")
        x = x.replace("[", ",")
        x = x.replace("]", ",")
        x = re.sub(r",+", ",", x)
        x_list = x.split(",")
        # strip all elements
        x_list = [i.strip() for i in x_list]
        # remove all elements that are just whitespace
        x_list = [i for i in x_list if len(i) > 2]
        return x_list

    def parse_predictions_in_gt(
        self, gts: list[str], pred: str, mode: str = "strict"
    ) -> list[bool]:
        """
        Determines whether the ground truth includes the prediction by assessing whether the
        string for the prediction is contained in any of the ground truth strings.

        mode = "strict" means every token from the prediction should be in at least one of the gt
        strings to be considered as a detection.

        Returns: list of boolean values indicating whether the prediction was found in the gt.
        """
        preds_in_gt = []
        gts = [self.preprocessing_gt(gt) for gt in gts]
        preds = self.preprocessing_pred_for_precision(pred)
        preds = list(set(preds))
        if mode == "strict":
            for pred in preds:
                is_in_gt = False
                for gt in gts:
                    if pred in gt:
                        is_in_gt = True
                        break
                preds_in_gt.append(is_in_gt)
        else:
            raise NotImplementedError

        assert len(preds) == len(
            preds_in_gt
        ), f"Length of predictions and  bool array of predictions in gt is not the same. {len(preds)} != {len(preds_in_gt)}"
        return preds_in_gt

    def get_parsed_bool_extraction_stats(
        self, gts: list[str], pred: str
    ) -> tuple[list]:

        gts = [self.preprocessing_gt(gt) for gt in gts]
        pred = self.preprocessing_gt(pred)
        gts_found = self.parse_ground_truths_in_prediction(gts, pred)
        preds_in_gt = self.parse_predictions_in_gt(gts, pred)
        return gts_found, preds_in_gt

    def get_sample_stats(self, results):
        all_preds_in_gt = []
        all_gts_found = []

        for subresult in results:
            gts_found, preds_in_gt = self.get_parsed_bool_extraction_stats(
                subresult["gold_list"], subresult["pred_list"]
            )
            all_gts_found.append(gts_found)
            all_preds_in_gt.extend(preds_in_gt)
        # create a single all_gts_found_list where element i is True if the ith element in any sub-list is True
        all_gts_found = np.array(all_gts_found)
        all_gts_found = np.any(all_gts_found, axis=0).tolist()
        # print(len(all_gts_found))
        assert len(all_gts_found) == len(
            gts_found
        ), f"Length of all_gts_found and all_preds_in_gt is not the same. {len(all_gts_found)} != {len(all_preds_in_gt)}"
        return all_gts_found, all_preds_in_gt
