from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

import os
from PIL import Image
import requests
import json
from tqdm import tqdm

images_dir = "/data/BADRI/RESEARCH/CIRCULARS/DATA/CIRCULARS_TESTSET/final/"
json_data = json.load(open("/data/BADRI/RESEARCH/CIRCULARS/DATA/CIRCULARS_TESTSET/final_annotations.json"))
results_dir = "/data/BADRI/RESEARCH/CIRCULARS/results/iter2/llava/v1/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16
)
model.to(device)
model.eval()



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


# image of a radar chart
        image = Image.open(image_path)
        prompt = f"[INST] <image>\n{question}[/INST]"

        inputs = processor(prompt, image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=100)
        
        result = processor.decode(output[0], skip_special_tokens=True)
        
        # remove from corresponding word
        word = "[/INST]"
        result = result[result.index(word):]
        # remove prefix
        word = "[/INST] "
        result = result[result.index(word) + len(word):]
        
        image_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": result
            })
        
            #save the results for each image to a json file
    with open(f"{results_dir}{image_filename}_results.json", "w") as f:
        json.dump(image_results, f, indent=2)
            
    results[image_filename] = image_results

# Save results to a JSON file
output_file = "/data/BADRI/RESEARCH/CIRCULARS/results/iter2/llava/v1_inference_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")
            
# The image appears to be a radar chart, which is a type of multivariate chart that displays values for multiple variables represented on axes
# starting from the same point. This particular radar chart is showing the performance of different models or systems across various metrics.
# The axes represent different metrics or benchmarks, such as MM-Vet, MM-Vet, MM-Vet, MM-Vet, MM-Vet, MM-V
