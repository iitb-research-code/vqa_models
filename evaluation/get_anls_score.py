from anls import anls_score


predictions = []
ground_truths = []



anls_score(prediction=predictions, gold_labels=[ground_truths], threshold=0.5)