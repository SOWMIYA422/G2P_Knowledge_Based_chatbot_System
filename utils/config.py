import os

# Dataset paths
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indian_rice_varieties.csv")

# Embedding model
PLANT_BERT_MODEL = "zhihan1996/PlantBERT"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS settings
BATCH_SIZE = 32
TOP_K = 5

# Prediction settings
TARGET_COLUMN = "Yield_per_plant"
NON_FEATURE_COLS = ["SampleID"]

# Cross-breeding
HYBRID_STRATEGY = "average"

# Knowledge Base Settings
KNOWLEDGE_RULES_PATH = "models/knowledge_rules.json"
ONTOLOGY_PATH = "models/ontology.json"

# Visualization
PLOTLY_THEME = "plotly_white"
