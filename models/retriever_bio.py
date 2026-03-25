import faiss
import pandas as pd
import numpy as np
import sys
import os
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.embeddings import PlantBERTEmbedder
from utils.config import DATASET_PATH

class BioRetriever:
    def __init__(self, dataset_path=DATASET_PATH):
        try:
            # Resolve dataset path
            resolved_path = self._resolve_dataset_path(dataset_path)
            logger.info(f"Loading dataset from: {resolved_path}")

            # Load dataset
            self.df = pd.read_csv(resolved_path)

            # Clean column names
            self.df.columns = (
                self.df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
            )

            logger.info(
                f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns"
            )
            logger.info(f"Columns: {list(self.df.columns)}")

            # Initialize embedder
            self.embedder = PlantBERTEmbedder()

            # FAISS index + embeddings
            self.index = None
            self.embeddings = None

            # Build the vector index
            self._build_index()

        except Exception as e:
            logger.error(f"Error initializing BioRetriever: {e}")
            raise

    def _resolve_dataset_path(self, dataset_path):
        """Resolve the dataset file path with multiple fallback options"""
        # Check if the provided path exists
        if os.path.exists(dataset_path):
            return dataset_path

        # Try different possible locations
        possible_locations = [
            dataset_path,
            os.path.join(os.path.dirname(__file__), "..", dataset_path),
            os.path.join(os.path.dirname(__file__), dataset_path),
            "dataset.csv",
            "../dataset.csv",
            os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"),
            os.path.join(os.path.dirname(__file__), "data", "dataset.csv"),
            "data/dataset.csv",
            "../data/dataset.csv",
        ]

        for location in possible_locations:
            if os.path.exists(location):
                logger.info(f"Found dataset at: {location}")
                return location

        # If no file found, show helpful error
        error_msg = f"""
❌ Dataset file not found!

Tried the following locations:
{chr(10).join(f"  • {loc}" for loc in possible_locations)}

Please make sure:
1. Your dataset.csv file exists in one of these locations
2. Or update the dataset_path parameter when creating BioRetriever

You can:
- Create a 'data' folder and put dataset.csv inside it
- Or specify the full path: BioRetriever('C:/path/to/your/dataset.csv')
"""
        raise FileNotFoundError(error_msg)

    def _build_index(self):
        """Build FAISS index from dataset embeddings"""
        try:
            # Map Genotype / Phenotype / Environment fields
            texts = []
            for _, row in self.df.iterrows():
                text_parts = []

                # Genotype information
                if "Variety" in row and pd.notna(row["Variety"]):
                    text_parts.append(f"Variety: {row['Variety']}")

                # Phenotype information
                phenotype_fields = [
                    "Yield_per_plant",
                    "Height",
                    "Grain_weight",
                    "Drought_Tolerance",
                ]
                phenotype_info = []
                for field in phenotype_fields:
                    if field in row and pd.notna(row[field]):
                        phenotype_info.append(f"{field}: {row[field]}")
                if phenotype_info:
                    text_parts.append("Phenotypes: " + ", ".join(phenotype_info))

                # Environment information
                env_fields = ["Rainfall_mm", "Temperature_C", "Soil_Type", "Country"]
                env_info = []
                for field in env_fields:
                    if field in row and pd.notna(row[field]):
                        env_info.append(f"{field}: {row[field]}")
                if env_info:
                    text_parts.append("Environment: " + ", ".join(env_info))

                # SNP information (first 5 SNPs as example)
                snp_cols = [c for c in self.df.columns if c.startswith("OsSNP")]
                if snp_cols:
                    snp_info = [
                        f"{col}={row[col]}"
                        for col in snp_cols[
                            :3
                        ]  # Show only first 3 to keep text manageable
                        if pd.notna(row[col])
                    ]
                    if snp_info:
                        text_parts.append("SNPs: " + ", ".join(snp_info))

                texts.append(". ".join(text_parts))

            logger.info(f"Generated {len(texts)} text representations for embedding")

            # Encode into embeddings
            if len(texts) == 0:
                raise ValueError("No valid text representations generated from dataset")

            self.embeddings = self.embedder.encode(texts, batch_size=16)
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")

            # Build FAISS index
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype("float32"))
            logger.info("✅ FAISS index built successfully")

        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise

    def query(self, query_text, top_k=5):
        """Query the knowledge base"""
        try:
            if self.index is None:
                raise ValueError("FAISS index not built. Call _build_index() first.")

            if not query_text or not query_text.strip():
                raise ValueError("Query text cannot be empty")

            # Encode query
            q_embed = self.embedder.encode([query_text]).astype("float32")

            # Search FAISS
            distances, indices = self.index.search(q_embed, top_k)

            # Return top_k matching rows
            results = self.df.iloc[indices[0]].copy()
            results["similarity_score"] = 1 / (
                1 + distances[0]
            )  # Convert distance to similarity
            return results

        except Exception as e:
            logger.error(f"Error in query: {e}")
            return pd.DataFrame()

    def get_dataset_info(self):
        """Get basic information about the loaded dataset"""
        if self.df is None:
            return "No dataset loaded"

        info = {
            "total_samples": len(self.df),
            "columns": list(self.df.columns),
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        }

        # Add some basic statistics for key columns
        if "Variety" in self.df.columns:
            info["unique_varieties"] = self.df["Variety"].nunique()

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in ["Yield_per_plant", "Height", "Grain_weight"]:
            if col in self.df.columns:
                info[f"{col}_mean"] = self.df[col].mean()
                info[f"{col}_std"] = self.df[col].std()

        return info

    def search_by_trait(self, trait_name, operator=">", value=None, top_k=5):
        """Search by specific trait values"""
        try:
            if trait_name not in self.df.columns:
                raise ValueError(f"Trait '{trait_name}' not found in dataset")

            if value is None:
                # If no value provided, return top values
                results = self.df.nlargest(top_k, trait_name)
            else:
                # Filter based on operator
                if operator == ">":
                    results = self.df[self.df[trait_name] > value].nlargest(
                        top_k, trait_name
                    )
                elif operator == ">=":
                    results = self.df[self.df[trait_name] >= value].nlargest(
                        top_k, trait_name
                    )
                elif operator == "<":
                    results = self.df[self.df[trait_name] < value].nlargest(
                        top_k, trait_name
                    )
                elif operator == "<=":
                    results = self.df[self.df[trait_name] <= value].nlargest(
                        top_k, trait_name
                    )
                elif operator == "==":
                    results = self.df[self.df[trait_name] == value].nlargest(
                        top_k, trait_name
                    )
                else:
                    raise ValueError(f"Unsupported operator: {operator}")

            return results

        except Exception as e:
            logger.error(f"Error in trait search: {e}")
            return pd.DataFrame()
