from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from config import *



image = Image.open("/data/BADRI/MISC/CIRCULARS/data/samples/sample.jpg")
question = "What is the Subject of the document?"

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to(DEVICE)
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")

inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE)
predictions = model.generate(**inputs)

print(processor.decode(predictions[0], skip_special_tokens=True))