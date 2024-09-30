import json
from evaluation_metrics import *


def get_results(reference, prediction):
    return evaluate(reference, prediction)


#Load json file
with open('/data/BADRI/RESEARCH/CIRCULARS/results/v2_inference_results.json', 'r') as f:
    data = json.load(f)
    

# make a list of reference and prediction
reference = []
prediction = []

# Json data
# {
#   "img_fp_pdfs_admin_2_2019_09_23_12_29_07.png": [
#     {
#       "question": "What is the title of the document?",
#       "ground_truth": "HIGH COURT FOR THE STATE OF TELANGANA.",
#       "predicted_answer": "The title of the document is \"HIGH COURT FOR THE STATE OF TELANGANA.\""
#     },
#     {
#       "question": "What is the reference number of this circular?",
#       "ground_truth": "ROC.NO. 465/2019-C1.",
#       "predicted_answer": "The reference number of this circular is 1."
#     },
#     {
#       "question": "What is the date of this circular?",
#       "ground_truth": "18.07.2019",
#       "predicted_answer": "The date of this circular is 12th July 2019."
#     },
#   ]
# }

for key, value in data.items():
    for item in value:
        reference.append(item['ground_truth'])
        prediction.append(item['predicted_answer'])

# #Top k results
# top_k = 5

# #Get top k results
# reference = reference[:top_k]
# prediction = prediction[:top_k]

results = get_results(reference, prediction)
print(results)