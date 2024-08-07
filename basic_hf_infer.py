from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from config import *

image = Image.open(SAMPLE_IMAGE)

model = Pix2StructForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
processor = Pix2StructProcessor.from_pretrained(MODEL)

inputs = processor(images=image, text=SAMPLE_QUESTION, return_tensors="pt").to(DEVICE)
predictions = model.generate(**inputs)

print(processor.decode(predictions[0], skip_special_tokens=True))