import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import logging

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

logger = logging.getLogger(__name__)

from utils.config import DATASET_PATH

class PhenotypePredictor:
    def __init__(self, dataset_path=DATASET_PATH):
        try:
            # Resolve dataset path
            resolved_path = self._resolve_dataset_path(dataset_path)
            logger.info(f"Loading dataset from: {resolved_path}")

            self.df = pd.read_csv(resolved_path)
            self.model = None
            self.feature_importance = None
            self.explainer = None
            self.feature_names = None
            self.category_mappings = {} # Store categories for consistency
            logger.info("✅ PhenotypePredictor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
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

        raise FileNotFoundError(f"Dataset file not found. Tried: {possible_locations}")

    def _encode_categoricals(self, df: pd.DataFrame, is_training: bool = False):
        """Convert categorical columns into numeric codes consistently"""
        df_encoded = df.copy()
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        # Only encode columns that are actually in the feature set
        cat_cols = [c for c in cat_cols if c in df_encoded.columns]
        
        for col in cat_cols:
            if is_training:
                # Store categories from the training set
                self.category_mappings[col] = pd.Categorical(self.df[col]).categories
            
            if col in self.category_mappings:
                # Apply stored categories to ensure consistent mapping
                df_encoded[col] = pd.Categorical(df_encoded[col], categories=self.category_mappings[col]).codes
            else:
                # Fallback to simple codes if not in training
                df_encoded[col] = df_encoded[col].astype("category").cat.codes
        return df_encoded

    def train(self, target="Yield_per_plant"):
        """Train XGBoost model to predict phenotypes"""
        try:
            # Drop metadata columns that should not be used as features
            non_feature_cols = ["SampleID", "Variety"]
            feature_cols = [
                col for col in self.df.columns if col not in non_feature_cols + [target]
            ]
            self.feature_names = feature_cols

            X = self.df[feature_cols].copy()
            y = self.df[target]

            # Encode categorical features and store mappings
            X = self._encode_categoricals(X, is_training=True)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # XGBoost regressor with optimized parameters
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="rmse",
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Initialize SHAP explainer
            if _HAS_SHAP:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    logger.info("✅ SHAP TreeExplainer initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize SHAP explainer: {e}")

            # Evaluate performance
            preds = self.model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)

            # Feature importance
            self.feature_importance = pd.DataFrame(
                {"feature": feature_cols, "importance": self.model.feature_importances_}
            ).sort_values("importance", ascending=False)

            logger.info(f"✅ Model trained on {target}. RMSE: {rmse:.4f}, R²: {r2:.4f}")
            print(f"✅ XGBoost trained on {target}. RMSE: {rmse:.4f}, R²: {r2:.4f}")

            return rmse, r2

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, input_data: dict):
        """Predict phenotype value given input genotype/environment data"""
        if self.model is None:
            raise ValueError("⚠️ Model is not trained yet. Call train() first.")

        try:
            # Ensure input data matches feature names
            features = {name: input_data.get(name, 0) for name in self.feature_names}
            input_df = pd.DataFrame([features])
            input_df = self._encode_categoricals(input_df, is_training=False)
            return self.model.predict(input_df)[0]
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def get_shap_explanation(self, input_data: dict):
        """Get SHAP values for a specific input"""
        if not _HAS_SHAP or self.explainer is None:
            return None, None

        try:
            # Prepare input data
            features = {name: input_data.get(name, 0) for name in self.feature_names}
            input_df = pd.DataFrame([features])
            input_df = self._encode_categoricals(input_df, is_training=False)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(input_df)
            expected_value = self.explainer.expected_value
            
            return shap_values[0], expected_value
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None, None

    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        return self.feature_importance.head(top_n)
