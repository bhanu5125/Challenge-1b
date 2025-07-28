# download_models.py
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from pathlib import Path
import logging

# --- Configuration ---
# Dictionary mapping a local-friendly name to the Hugging Face model identifier
MODELS_TO_DOWNLOAD = {
    "bert-tiny": "prajjwal1/bert-tiny",
    "t5-small": "t5-small"  # <-- CHANGED: Replaced distilbart with t5-small
}
# Define the local directory to save the models
LOCAL_MODELS_DIR = Path("./models")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_models():
    """
    Downloads and saves specified Hugging Face models and their tokenizers locally.
    """
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {LOCAL_MODELS_DIR.resolve()}")

    for local_name, hub_name in MODELS_TO_DOWNLOAD.items():
        save_directory = LOCAL_MODELS_DIR / local_name
        
        # Check if the model directory already exists and has content
        if save_directory.exists() and any(save_directory.iterdir()):
             logger.info(f"Model '{local_name}' already seems to exist in {save_directory}. Skipping download.")
             continue

        logger.info(f"Downloading model '{hub_name}' to '{save_directory}'...")
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hub_name)
            tokenizer.save_pretrained(save_directory)
            logger.info(f"  - Tokenizer for '{hub_name}' downloaded successfully.")

            # Determine the correct AutoModel class
            if "bart" in hub_name or "t5" in hub_name:
                model = AutoModelForSeq2SeqLM.from_pretrained(hub_name)
            else:
                model = AutoModel.from_pretrained(hub_name)
            
            # Download and save the model weights
            model.save_pretrained(save_directory)
            logger.info(f"  - Model weights for '{hub_name}' downloaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to download or save model '{hub_name}'. Error: {e}")

if __name__ == "__main__":
    download_models()
    logger.info("✅ All models processed.")
