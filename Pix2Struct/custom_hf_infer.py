import json
import os
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from config import *

model = Pix2StructForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
processor = Pix2StructProcessor.from_pretrained(MODEL)

with open(JSON_FILE) as f:
    json_data = json.load(f)
    
# for data_point in tqdm(qna_data):
#     image = Image.open(IMAGES_DIR + data_point['file_name'])
#     for qna in tqdm(data_point['question_answer_pairs']):
        
#         inputs = processor(images=image, text=qna['question'], return_tensors="pt").to(DEVICE)
#         predictions = model.generate(**inputs, max_length=512)
#         qna['pred_answer'] = processor.decode(predictions[0], skip_special_tokens=True)
        
#     # #Dynamic updation of Json file
#     with open('temp.json', 'w') as f:
#         json.dump(qna_data, f, indent=4, ensure_ascii=False, sort_keys=True)

results = {}

results_dir = "/data/BADRI/RESEARCH/CIRCULARS/results/pix2struct/"
images_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/testset/"


for image_filename, qa_pairs in json_data.items():
    image_path = os.path.join(images_dir, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    image_results = []
    
    #read image
    image = Image.open(image_path)
    
    for qa_pair in tqdm(qa_pairs):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]
        
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE)
        predictions = model.generate(**inputs, max_length=128)
        qa_pair['pred_answer'] = processor.decode(predictions[0], skip_special_tokens=True)
        
        image_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "pred_answer": qa_pair['pred_answer']
        })
        
    results[image_filename] = image_results
    
     #save the results for each image to a json file
    with open(f"{results_dir}{image_filename}_results.json", "w") as f:
        json.dump(image_results, f, indent=2)

# save json
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

