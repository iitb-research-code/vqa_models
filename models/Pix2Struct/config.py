

DEVICE = 1
MODEL = "google/pix2struct-docvqa-base"
PROCESSOR_BASE = "ybelkada/pix2struct-base"

MODEL_FILE = "/data/BADRI/MISC/CIRCULARS/models/pix2struct-model/"

SAMPLE_IMAGE = "/data/BADRI/MISC/CIRCULARS/data/samples/sample.jpg"
SAMPLE_QUESTION = "What is the Subject of the document?"


# IMAGES_DIR = "/data/BADRI/MISC/CIRCULARS/data/batch1/images/"
# JSON_FILE = "/data/BADRI/MISC/CIRCULARS/data/data_final.json"

IMAGES_DIR = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/testset/"
JSON_FILE = "/data/BADRI/RESEARCH/CIRCULARS/doc_vlm_application/data/all_qna_pairs.json"
RESULTS_FILE = "/data/BADRI/RESEARCH/CIRCULARS/results/pix2struct/all_qna_pairs_results.json"


# Hyperparamters
MAX_PATCHES = 1024
LEARNING_RATE = 1e-4
EPOCHS = 50

