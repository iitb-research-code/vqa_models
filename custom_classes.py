from torch.utils.data import Dataset
from PIL import Image

class Pix2StructDataset(Dataset):
    def __init__(self, data, processor, max_patches):
        self.data = data
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['document'])
        except:
            return None
        processed_data = self.processor(images=image, return_tensors="pt", text=item["question"], max_patches=self.max_patches)
        encoding = {}
        for key in processed_data.keys():
            if key in ['flattened_patches', 'attention_mask']:
                encoding[key] = processed_data[key].squeeze()
        encoding['answer'] = item['answer']
        encoding['question'] = item['question']
        encoding['document'] = item['document']
        # print("Encoding Keys", encoding.keys())
        return encoding