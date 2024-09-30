from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import random
from tqdm import tqdm
import os


device = "cuda:0"

results_dir = "/data/BADRI/RESEARCH/CIRCULARS/results/"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

model.to(device)
model.bfloat16()
model.eval()

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


def custom_hf_inference(messages):
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",)

    # Ensure all inputs are moved to the same device

    inputs = inputs.to(device)


    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text


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
            predicted_answer = output[0] if output else ""
            
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
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
            

