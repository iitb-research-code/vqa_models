from transformers import pipeline
import pytesseract
from PIL import Image
import json
import os
from tqdm import tqdm


def perform_ocr(image_path):
    image = Image.open(image_path)
    result = pytesseract.image_to_string(image_path)
    return result

def load_llm_model(device):
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)
    return pipe

def generate_llm_answer(question, context, pipe):
    
    content = f"Question: {question}\n\nContext: {context}\n\n Give your answer in a crisp manner. Do not add any preamble or postamble."
    messages = [ {"role": "user", "content": content}]
    result = pipe(messages, max_new_tokens=64, do_sample=True, temperature=0.7)
    # print(result[0]["generated_text"][1])
    # exit()
    ans = result[0]["generated_text"][1]['content']
    # print(question)
    # print(ans)
    return ans



images_dir = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/images/final_testset/"
json_data = json.load(open("/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/all_qna_pairs.json"))

results_dir = "/data/BADRI/RESEARCH/CIRCULARS/results/llama/"
output_file = "/data/BADRI/RESEARCH/CIRCULARS/results/llama/results.json"


device = "cuda"
pipe = load_llm_model(device)

# run inference for each image
results = {}

for image_filename, qa_pairs in json_data.items():
    image_path = os.path.join(images_dir, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    image_results = []
    
    ocr_output = perform_ocr(image_path)
    
    for qa_pair in tqdm(qa_pairs):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]


        try:
            output = generate_llm_answer(question, ocr_output, pipe)
            predicted_answer = output if output else ""
            
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


with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
            