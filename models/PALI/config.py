

DEVICE = 1
MODEL = "google/paligemma-3b-ft-docvqa-224"
PROCESSOR_BASE = "google/paligemma-3b-ft-docvqa-224"

MODEL_FILE = "/data/BADRI/MISC/CIRCULARS/models/pix2struct-model/"

SAMPLE_IMAGE = "/data/BADRI/MISC/CIRCULARS/data/samples/sample.jpg"
SAMPLE_QUESTION = "What is the date of the document?"


IMAGES_DIR = "/data/BADRI/MISC/CIRCULARS/data/batch1/images/"
JSON_FILE = "/data/BADRI/MISC/CIRCULARS/data/data_final.json"

RESULTS_FILE = "/data/BADRI/MISC/CIRCULARS/data/data_final_fintune.json"


# Hyperparamters
MAX_PATCHES = 1024
LEARNING_RATE = 1e-4
EPOCHS = 50

