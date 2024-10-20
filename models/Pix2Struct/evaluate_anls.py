import numpy as np
import json
from Levenshtein import distance as levenshtein_distance

from config import RESULTS_FILE

def normalized_levenshtein_similarity(pred, truth):
    """
    Calculate the normalized Levenshtein similarity between two strings.
    
    :param pred: Predicted string.
    :param truth: Ground truth string.
    :return: Normalized Levenshtein similarity (1 - normalized edit distance).
    """
    len_max = max(len(pred), len(truth))
    if len_max == 0:
        return 1.0  # If both strings are empty, similarity is 1.
    
    lev_dist = levenshtein_distance(pred, truth)
    norm_similarity = 1 - (lev_dist / len_max)
    return norm_similarity

def evaluate_anls(pred_answers, ground_truth_answers):
    """
    Evaluate the ANLS score for a dataset of predicted and ground truth answers.
    
    :param pred_answers: List of strings, where each string is a predicted answer.
    :param ground_truth_answers: List of lists of strings, where each list contains possible ground truth answers.
    :return: The ANLS score for the entire dataset.
    """
    similarities = []
    
    for pred, ground_truth_list in zip(pred_answers, ground_truth_answers):
        # Calculate normalized Levenshtein similarity for each ground truth and take the maximum
        max_similarity = max(normalized_levenshtein_similarity(pred, truth) for truth in ground_truth_list)
        similarities.append(max_similarity)
    
    # Average the similarities across the dataset
    anls_score = np.mean(similarities)
    return anls_score

# with open(RESULTS_FILE) as f:
#     data = json.load(f)

#Load json file
with open('/data/BADRI/RESEARCH/CIRCULARS/results/v1_inference_results.json', 'r') as f:
    data = json.load(f)
    
reference = []
prediction = []


for key, value in data.items():
    for item in value:
        reference.append(item['ground_truth'])
        prediction.append(item['predicted_answer'])
    
# for file in data:
#     # print(file["file_name"])
#     for qa in file["question_answer_pairs"]:
#         try:
#             pred_answers.append(qa["pred_answer"])
#             ground_truth_answers.append([qa["answer"]])
#         except:
#             print(qa)
#             exit()

anls_score = evaluate_anls(prediction, reference)
print(f"ANLS Score: {anls_score:.4f}")