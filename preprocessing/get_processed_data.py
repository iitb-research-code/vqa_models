import json
import os
import re
import shutil


images_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/images/final_testset/"


json_file = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/annotation_badri.json"

# Function to remove redundant parts from questions
def remove_redundant_parts(question):
    return re.sub(r"Q\d+: \d+\. ", "", question)

# Function to remove redundant parts from answers
def remove_redundant_parts_from_answer(answer):
    return re.sub(r"^A\d+: :\s*", "", answer)

#read json file
with open(json_file, 'r') as f:
    data = json.load(f)


# Dictionary to accumulate Q&A pairs for each image
all_qna_pairs = {}

testset_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/testset/"
image_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/images/final_testset/"

# Iterate through the JSON data
for image, annotations in data.items():
    # Initialize a list to store Q&A pairs for the current image
    image_qna_pairs = []
    
    # if image is present in the image dir, copy it the test_images folder
    if os.path.exists(os.path.join(images_dir, image)):
        shutil.copy(os.path.join(images_dir, image), os.path.join("testset", image))
        
        
    
    for id, annotation in annotations.items():
        try:
            if annotation['Type'] == 'Extractive':
                qna_pair = annotation['q_a_pair']
            
            # Use edited question/answer if not empty, otherwise use original
            question = qna_pair['edited_question'] if qna_pair['edited_question'] else qna_pair['original_question']
            answer = qna_pair['edited_answer'] if qna_pair['edited_answer'] else qna_pair['original_answer']
            
            # Remove redundant parts from the question
            question = remove_redundant_parts(question)
            
            # Remove redundant parts from the answer
            answer = remove_redundant_parts_from_answer(answer)
            
            image_qna_pairs.append({
                'question': question,
                'answer': answer,
            })
        except Exception as e:
            print(e)
    
    if image_qna_pairs:
        all_qna_pairs[image] = image_qna_pairs

print("saving data")

# Save all Q&A pairs to a single file
output_file = os.path.join("all_qna_pairs.json")
with open(output_file, 'w') as out_f:
    json.dump(all_qna_pairs, out_f, indent=4)

