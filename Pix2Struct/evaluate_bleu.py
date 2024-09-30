from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from config import *
import numpy as np


def evaluate_bleu_on_dataset(pred_answers, ground_truth_answers, metric='average'):
    """
    Evaluate BLEU score for a dataset of predicted and ground truth answers.

    :param pred_answers: List of strings, where each string is a predicted answer.
    :param ground_truth_answers: List of lists of strings, where each list contains possible ground truth answers.
    :param metric: Metric to calculate on the BLEU scores ('average' or 'median').
    :return: The average or median BLEU score for the entire dataset.
    """
    bleu_scores = []
    smooth_fn = SmoothingFunction().method1  # Smoothing function for BLEU score
    
    for pred, ground_truth in zip(pred_answers, ground_truth_answers):
        # Tokenize the sentences
        pred_tokens = pred.split()
        ground_truth_tokens = [ans.split() for ans in ground_truth]
        
        # Calculate BLEU score
        bleu_score = sentence_bleu(ground_truth_tokens, pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu_score)
    
    if metric == 'average':
        return np.mean(bleu_scores)
    elif metric == 'median':
        return np.median(bleu_scores)
    else:
        raise ValueError("Metric must be 'average' or 'median'.")
    
    
def evaluate_bleu(pred_answers, ground_truth_answers):
    """
    Evaluate BLEU score for a list of predicted answers and ground truth answers.

    :param pred_answers: List of strings, where each string is a predicted answer.
    :param ground_truth_answers: List of lists of strings, where each list contains possible ground truth answers.
    :return: List of BLEU scores for each pair of predicted and ground truth answers.
    """
    bleu_scores = []
    smooth_fn = SmoothingFunction().method1  # Smoothing function for BLEU score
    
    for pred, ground_truth in zip(pred_answers, ground_truth_answers):
        # Tokenize the sentences
        pred_tokens = pred.split()
        ground_truth_tokens = [ans.split() for ans in ground_truth]
        
        # Calculate BLEU score
        bleu_score = sentence_bleu(ground_truth_tokens, pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu_score)
    
    return bleu_scores

# with open(RESULTS_FILE) as f:
#     data = json.load(f)
    
# pred_answers = []
# ground_truth_answers = []

    
# for file in data:
#     # print(file["file_name"])
#     for qa in file["question_answer_pairs"]:
#         try:
#             pred_answers.append(qa["pred_answer"])
#             ground_truth_answers.append([qa["answer"]])
#         except:
#             print(qa)
#             exit()

#Load json file
with open('/data/BADRI/RESEARCH/CIRCULARS/results/v1_inference_results.json', 'r') as f:
    data = json.load(f)
    
reference = []
prediction = []


for key, value in data.items():
    for item in value:
        reference.append(item['ground_truth'])
        prediction.append(item['predicted_answer'])
        
pred_answers = prediction
ground_truth_answers = reference
    

# bleu_scores = evaluate_bleu(pred_answers, ground_truth_answers)
# for i, score in enumerate(bleu_scores):
#     print(f"Q{i+1} BLEU Score: {score:.4f}")

average_bleu_score = evaluate_bleu_on_dataset(pred_answers, ground_truth_answers, metric='average')
median_bleu_score = evaluate_bleu_on_dataset(pred_answers, ground_truth_answers, metric='median')

print(f"Average BLEU Score: {average_bleu_score:.4f}")
print(f"Median BLEU Score: {median_bleu_score:.4f}")
