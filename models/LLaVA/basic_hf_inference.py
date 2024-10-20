

# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import os
import json


processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.6-vicuna-7b")
model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.6-vicuna-7b")


# Run inference
def custom_hf_inference(messages):
    inputs = processor(messages, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=64)
    return processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# message = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": image_path},
#             {"type": "text", "text": question},
#         ],
#     }
# ]

images_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/images/final_testset/"

json_data = json.load(open("/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/all_qna_pairs.json"))

results = {}

for image_filename, qa_pairs in json_data.items():
    image_path = os.path.join(images_dir, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    image_results = []
    
    for qa_pair in tqdm(qa_pairs):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]
        
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                    {"type": "text", "text": "Give your answer in a crisp manner. Do not add any preamble or postamble."}
                ],
            }
        ]
        
        try:
            output = custom_hf_inference(message)
            # print(output)
            image_results.append({"question": question, "answer": output})
        except Exception as e:
            print(f"Error processing image: {image_path}, question: {question}")
            print(e)
    
    results[image_filename] = image_results