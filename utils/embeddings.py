import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback library
try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.warning("sentence-transformers not available, using fallback")


class PlantBERTEmbedder:
    def __init__(
        self,
        preferred_model="zhihan1996/PlantBERT",
        fallback_model="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    ):
        """
        Try preferred_model (transformers). If it fails (not found / gated),
        automatically fall back to a SentenceTransformer model.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = None
        self.use_st = False

        # Try preferred transformers model
        try:
            logger.info(f"Loading preferred model: {preferred_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(preferred_model)
            self.model = AutoModel.from_pretrained(preferred_model)
            self.model.to(self.device)
            self.model_name = preferred_model
            self.use_st = False
            logger.info(f"✅ Successfully loaded transformers model: {preferred_model}")

        except Exception as e:
            # Log the error, then try fallback
            logger.warning(f"Could not load preferred model '{preferred_model}': {e}")
            logger.info(f"Falling back to SentenceTransformer '{fallback_model}'")

            if not _HAS_ST:
                raise RuntimeError(
                    "sentence-transformers is required as fallback. Install it: pip install sentence-transformers"
                )

            # Load SentenceTransformer fallback
            self.st_model = SentenceTransformer(fallback_model, device=self.device)
            self.model_name = fallback_model
            self.use_st = True
            logger.info(f"✅ Successfully loaded fallback model: {fallback_model}")

    def encode(self, texts, batch_size=16):
        """
        Encode texts -> numpy array of embeddings.
        Supports either transformers AutoModel (mean pooling) or SentenceTransformer.
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.use_st:
            # SentenceTransformer path (fast, single call)
            emb = self.st_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.array(emb)

        # transformers AutoModel path (manual batching + mean pooling)
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens)
            batch_emb = output.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_emb)

        return np.vstack(embeddings)

    def get_model_info(self):
        """Return information about the current model"""
        return {
            "model_name": self.model_name,
            "using_sentence_transformers": self.use_st,
            "device": self.device,
        }
