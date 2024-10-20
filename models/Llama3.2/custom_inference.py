import requests
import torch
import json
import os
from tqdm import tqdm
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

device = "cuda"

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.to(device)
model.bfloat16()
model.eval()

processor = AutoProcessor.from_pretrained(model_id)


def run_inference(image_path, question):
    image = Image.open(image_path)
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
            {"type": "text", "text": "Give your answer in a crisp manner. Do not add any preamble or postamble."}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=64)
    
    result = processor.decode(output[0])
    
    # Extract the assistant's answer
    start_token = "<|start_header_id|>assistant<|end_header_id|>"
    end_token = "<|eot_id|>"
    start_index = result.find(start_token) + len(start_token)
    result = result[start_index:]
    
    result = result[:result.find(end_token)]
    
    #remove new lines /n/n
    result = result.replace("\n\n", "")
    
    return result



images_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/images/final_testset/"
json_data = json.load(open("/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/all_qna_pairs.json"))

images_dir = "/data/BADRI/RESEARCH/CIRCULARS/DATA/CIRCULARS_TESTSET/final/"
json_data = json.load(open("/data/BADRI/RESEARCH/CIRCULARS/DATA/CIRCULARS_TESTSET/final_annotations.json"))

results_dir = "/data/BADRI/RESEARCH/CIRCULARS/results/iter2/llama32/v2/"
output_file = "/data/BADRI/RESEARCH/CIRCULARS/results/iter2/llama32/v2_inference_results.json"

# run inference for each image
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

        try:
            predicted_answer = run_inference(image_path, question)
            # predicted_answer = output[0] if output else ""
            # print(predicted_answer)
            # exit()
            
            image_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer
            })
        except Exception as e:
            print(f"Error processing {image_filename} with question '{question}': {str(e)}")
            
    #save the results for each image to a json file
    with open(f"{results_dir}{image_filename}_results.json", "w") as f:
        json.dump(image_results, f, indent=2)
            
    results[image_filename] = image_results

# Save results to a JSON file
output_file = "inference_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")
            
