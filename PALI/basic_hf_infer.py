from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# Load model directly
from transformers import AutoProcessor, AutoModelForPreTraining


from config import *

image = Image.open(SAMPLE_IMAGE)

processor = AutoProcessor.from_pretrained(PROCESSOR_BASE)
model = AutoModelForPreTraining.from_pretrained(MODEL).to(DEVICE)


inputs = processor(images=image, text=SAMPLE_QUESTION, return_tensors="pt").to(DEVICE)
predictions = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(predictions[0], skip_special_tokens=True))