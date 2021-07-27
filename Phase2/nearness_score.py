from typing import List
from re import findall


def compute_nearness_signs(
    tokenized_references: List[str], tokenized_candidates: List[str]
):
    correct_matches = 0
    incorrect_matches = 0
    for ground_truth, predicted in zip(tokenized_references, tokenized_candidates):
        ground_truth_signs = "".join(findall("[\+\-]+", ground_truth))
        predicted_signs = "".join(findall("[\+\-]+", predicted))
        if ground_truth_signs == predicted_signs:
            correct_matches += 1
        else:
            incorrect_matches += 1
    return correct_matches * 100 / (correct_matches + incorrect_matches)
