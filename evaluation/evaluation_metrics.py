import re
import string
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import evaluate
from Levenshtein import distance
bleu = evaluate.load("bleu")
from anls import anls_score

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s.strip().split("\n")[0]))))

def f1_score(prediction_, ground_truth):
    prediction_tokens = normalize_answer(prediction_).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def jacard(prediction_, ground_truth):
    a = set(prediction_.lower().split()) 
    b = set(ground_truth.lower().split())
    c = a.intersection(b)
    return float(len(c)) / abs(len(a) + len(b) - len(c))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction_, ground_truths):
    score = metric_fn(str(prediction_),str(ground_truths))
    return score


def get_rouge_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, ground_truth)
    return scores['rougeL'].fmeasure

def get_bleu_score(prediction, ground_truth):
    predictions = [prediction]
    references = [ground_truth]
    results = bleu.compute(predictions=predictions, references=references)
    return results['bleu']

def get_anls_score(reference, hypothesis):
    anls_score_val = anls_score(prediction=hypothesis, gold_labels=[reference], threshold=0.5)
    return anls_score_val


def evaluate(reference, prediction):
    exact_match = f1 = jacard_sim = rouge_score = bleu_score = anls_score = total = 0
    for i in range(len(reference)):
        ground_truths = reference[i]
        prediction_ = prediction[i]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction_, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction_, ground_truths)
        jacard_sim += jacard(prediction_, ground_truths)
        rouge_score += get_rouge_score(prediction_, ground_truths)
        bleu_score += get_bleu_score(prediction_, ground_truths)
        anls_score += get_anls_score(prediction_, ground_truths)
        total=total+1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    jacard_sim = 100.0 * jacard_sim / total
    rouge_score = 100.0 * rouge_score / total
    bleu_score = 100.0 * bleu_score / total
    anls_score = 100.0 * anls_score / total
    return {"exact_match": exact_match, "f1": f1, "jacard":jacard_sim, "rouge":rouge_score, "bleu":bleu_score, "anls":anls_score}

