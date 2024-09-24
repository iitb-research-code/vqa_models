import json
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from config import *

model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_FILE).to(DEVICE)
processor = Pix2StructProcessor.from_pretrained(MODEL)

# load model from pt file


with open(JSON_FILE) as f:
    qna_data = json.load(f)
    
for data_point in tqdm(qna_data):
    image = Image.open(IMAGES_DIR + data_point['file_name'])
    for qna in tqdm(data_point['question_answer_pairs']):
        
        inputs = processor(images=image, text=qna['question'], return_tensors="pt").to(DEVICE)
        predictions = model.generate(**inputs, max_length=512)
        qna['pred_answer'] = processor.decode(predictions[0], skip_special_tokens=True)
        
    # #Dynamic updation of Json file
    # with open('temp.json', 'w') as f:
    #     json.dump(qna_data, f, indent=4, ensure_ascii=False, sort_keys=True)
        

# save json
with open('/data/BADRI/MISC/CIRCULARS/data/data_final_fintune.json', 'w') as f:
    json.dump(qna_data, f, indent=4, ensure_ascii=False, sort_keys=True)
