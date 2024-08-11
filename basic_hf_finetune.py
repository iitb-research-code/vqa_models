import json
import torch

import warnings
warnings.filterwarnings("ignore")


from transformers import AutoProcessor, Pix2StructForConditionalGeneration

from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split

from config import *
from custom_classes import Pix2StructDataset


def collator(batch):
  # print("Collating")
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["answer"] for item in batch]

  documents = [item["document"] for item in batch]
  
  text_inputs = processor_base(text=texts, padding="max_length", return_tensors="pt", max_length=128, truncation=True)

  # Decode the text inputs
  # Getting back the text inputs -> Here
  # print("Text Inputs", processor_base.batch_decode(text_inputs.input_ids))
  
  new_batch["labels"] = text_inputs.input_ids
  
  for item in batch:
    # print("Item Keys", item.keys())
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])
  
  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"]) 
  new_batch["document"] = documents

  return new_batch


processor = AutoProcessor.from_pretrained(MODEL)
model = Pix2StructForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)

with open('/data/BADRI/MISC/CIRCULARS/data/dataset.json') as f:
    data_final = json.load(f)


processor_base = AutoProcessor.from_pretrained(PROCESSOR_BASE)


print("Data Generation Starting...")

# data_final = data_final[:]
dataset = Pix2StructDataset(data_final, processor, max_patches=MAX_PATCHES)

# Remove all the None type entries from the dataset
dataset = [item for item in dataset if item is not None]

print(dataset)

# Train - Test Split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=2, collate_fn=collator)
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=2, collate_fn=collator)

print("Data Generation Complete")



model.to(DEVICE)
model.train()


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Output examples, write to a seperate file
output_file = open("output_1.txt", "w")


for epoch in range(EPOCHS):
    train_loss = 0
    model.train()
    for idx, batch in enumerate(train_dataloader):
        labels = batch["labels"].to(DEVICE)
        flattened_patches = batch["flattened_patches"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    print("Epoch", epoch, "Training Loss", train_loss/len(train_dataloader))

    output_file.write("Epoch: " + str(epoch) + "\n")
    
    # Calculate the validation loss and ROUGE scores
    with torch.no_grad():
        validation_loss = 0
        total_rouge1 = 0
        total_rougeL = 0
        total_samples = 0
        for idx_1, batch_1 in enumerate(test_dataloader):
            documents = batch_1["document"]
            labels = batch_1["labels"].to(DEVICE)
            flattened_patches = batch_1["flattened_patches"].to(DEVICE)
            attention_mask = batch_1["attention_mask"].to(DEVICE)

            outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            validation_loss += loss.item()

            # Decode predictions and labels
            predictions = processor_base.batch_decode(outputs.logits.argmax(-1))
            labels = processor_base.batch_decode(labels)

            # Calculate ROUGE scores
            for pred, label in zip(predictions, labels):
                eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                eos_label = label.index("</s>") if "</s>" in label else len(label)

                scores = scorer.score(pred[:eos_pred], label[:eos_label])
                total_rouge1 += scores['rouge1'].fmeasure
                total_rougeL += scores['rougeL'].fmeasure
                total_samples += 1

            # Write predictions and labels to file
            for doc, pred, label in zip(documents, predictions, labels):
                eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                eos_label = label.index("</s>") if "</s>" in label else len(label)

                output_file.write("Document: " + str(doc) + "\n")
                output_file.write("Predictions: " + str(pred[:eos_pred]) + "\n")
                output_file.write("Labels: " + str(label[:eos_label]) + "\n")

        avg_rouge1 = total_rouge1 / total_samples
        avg_rougeL = total_rougeL / total_samples
        
        
    output_file.write("=============================================\n")
    print("Validation Loss", validation_loss/len(test_dataloader))
    print("Average ROUGE-1:", avg_rouge1)
    print("Average ROUGE-L:", avg_rougeL)


# save model
model.save_pretrained("/data/BADRI/MISC/CIRCULARS/models/pix2struct-model")

#save model through torch
torch.save(model, "/data/BADRI/MISC/CIRCULARS/models/model.pt")