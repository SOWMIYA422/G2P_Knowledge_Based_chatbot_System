import pandas as pd
import json
import numpy as np
from typing import Dict, List, Any
import os
import logging

logger = logging.getLogger(__name__)


from utils.config import DATASET_PATH

class PlantBreedingKnowledgeBase:
    def __init__(self, dataset_path=DATASET_PATH):
        try:
            # Resolve dataset path
            resolved_path = self._resolve_dataset_path(dataset_path)
            logger.info(f"Loading dataset from: {resolved_path}")

            self.df = pd.read_csv(resolved_path)
            self.rules = self._initialize_rules()
            self.ontology = self._build_ontology()
            logger.info("✅ Knowledge Base initialized with rules and ontology")
        except Exception as e:
            logger.error(f"Error initializing Knowledge Base: {e}")
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

    def _initialize_rules(self):
        """Expert rules for plant breeding recommendations"""
        return {
            "drought_tolerance_rules": [
                {
                    "condition": "Drought_Tolerance == 1",
                    "recommendation": "Suitable for arid regions with low rainfall",
                    "confidence": 0.95,
                    "action": "recommend_for_drought_areas",
                },
                {
                    "condition": "Rainfall_mm < 1000",
                    "recommendation": "Optimal for low rainfall areas (<1000mm)",
                    "confidence": 0.88,
                    "action": "suggest_water_efficient",
                },
            ],
            "yield_optimization_rules": [
                {
                    "condition": "Yield_per_plant > 35",
                    "recommendation": "High yield potential - suitable for commercial farming",
                    "confidence": 0.92,
                    "action": "recommend_high_yield",
                },
                {
                    "condition": "Grain_weight > 28",
                    "recommendation": "Excellent grain quality and weight",
                    "confidence": 0.87,
                    "action": "suggest_quality_focus",
                },
            ],
            "environment_adaptation_rules": [
                {
                    "condition": "Temperature_C > 25",
                    "recommendation": "Heat tolerant variety - suitable for warm climates",
                    "confidence": 0.85,
                    "action": "recommend_warm_climate",
                },
                {
                    "condition": "Soil_Type in ['Sandy', 'sandy']",
                    "recommendation": "Well-drained soil specialist",
                    "confidence": 0.82,
                    "action": "suggest_sandy_soil",
                },
                {
                    "condition": "Height > 100",
                    "recommendation": "Tall variety - consider wind exposure",
                    "confidence": 0.78,
                    "action": "warn_height_consideration",
                },
            ],
            "genetic_diversity_rules": [
                {
                    "condition": "Heterozygosity > 0.2",
                    "recommendation": "High genetic diversity - good for breeding programs",
                    "confidence": 0.90,
                    "action": "suggest_breeding_parent",
                }
            ],
        }

    def _build_ontology(self):
        """Domain ontology for plant breeding"""
        return {
            "traits": {
                "yield_related": ["Yield_per_plant", "Grain_weight"],
                "stress_tolerance": ["Drought_Tolerance", "Temperature_C"],
                "growth_characteristics": ["Height", "Coverage"],
                "genetic_diversity": ["Heterozygosity"],
                "environmental_adaptation": ["Rainfall_mm", "Soil_Type", "Country"],
            },
            "gene_functions": {
                "OsSNP001": "Drought response and water use efficiency",
                "OsSNP002": "Yield potential and grain development",
                "OsSNP003": "Root architecture and nutrient uptake",
                "OsSNP004": "Photosynthesis efficiency and biomass",
                "OsSNP005": "Disease resistance and plant immunity",
                "OsSNP006": "Flowering time and maturity",
                "OsSNP007": "Seed quality and storage proteins",
                "OsSNP008": "Stress response signaling",
                "OsSNP009": "Plant height and structure",
                "OsSNP010": "Nutrient use efficiency",
            },
            "breeding_strategies": {
                "pyramiding": "Combine multiple favorable alleles from different parents",
                "backcrossing": "Transfer specific traits to elite genetic background",
                "heterosis": "Exploit hybrid vigor through specific genetic combinations",
                "marker_assisted": "Use molecular markers for precise trait selection",
            },
            "environment_classes": {
                "arid": {
                    "rainfall": "<600mm",
                    "recommendation": "Drought tolerant varieties",
                },
                "semi_arid": {
                    "rainfall": "600-1000mm",
                    "recommendation": "Medium duration varieties",
                },
                "humid": {
                    "rainfall": ">1000mm",
                    "recommendation": "High yield potential varieties",
                },
            },
        }

    def infer_recommendations(self, genotype_data: Dict) -> List[Dict]:
        """Rule-based inference engine"""
        recommendations = []

        # Evaluate each rule category
        for category, rules in self.rules.items():
            for rule in rules:
                try:
                    if self._evaluate_condition(genotype_data, rule["condition"]):
                        recommendations.append(
                            {
                                "category": category.replace("_rules", "").title(),
                                "recommendation": rule["recommendation"],
                                "confidence": rule["confidence"],
                                "action": rule["action"],
                                "reasoning": self._generate_reasoning(
                                    genotype_data, rule
                                ),
                                "supporting_data": self._extract_supporting_data(
                                    genotype_data, rule
                                ),
                            }
                        )
                except Exception as e:
                    logger.warning(f"Rule evaluation failed: {e}")
                    continue

        # Sort by confidence and remove duplicates
        unique_recommendations = []
        seen_recommendations = set()
        for rec in sorted(recommendations, key=lambda x: x["confidence"], reverse=True):
            rec_key = rec["recommendation"][:100]  # Use first 100 chars as key
            if rec_key not in seen_recommendations:
                unique_recommendations.append(rec)
                seen_recommendations.add(rec_key)

        return unique_recommendations[:10]  # Return top 10

    def _evaluate_condition(self, data: Dict, condition: str) -> bool:
        """Evaluate rule conditions against genotype data"""
        try:
            # Handle numeric comparisons
            if (
                "Drought_Tolerance == 1" in condition
                and data.get("Drought_Tolerance") == 1
            ):
                return True
            if (
                "Yield_per_plant > 35" in condition
                and data.get("Yield_per_plant", 0) > 35
            ):
                return True
            if "Rainfall_mm < 1000" in condition and data.get("Rainfall_mm", 0) < 1000:
                return True
            if "Grain_weight > 28" in condition and data.get("Grain_weight", 0) > 28:
                return True
            if "Temperature_C > 25" in condition and data.get("Temperature_C", 0) > 25:
                return True
            if "Height > 100" in condition and data.get("Height", 0) > 100:
                return True
            if (
                "Heterozygosity > 0.2" in condition
                and data.get("Heterozygosity", 0) > 0.2
            ):
                return True
            if "Soil_Type" in condition and "sandy" in condition.lower():
                soil_type = str(data.get("Soil_Type", "")).lower()
                if "sandy" in soil_type:
                    return True

            return False
        except Exception as e:
            logger.warning(f"Condition evaluation error: {e}")
            return False

    def _generate_reasoning(self, data: Dict, rule: Dict) -> str:
        """Generate natural language reasoning for recommendations"""
        condition = rule["condition"]

        if "Drought_Tolerance" in condition:
            return f"Drought tolerance score: {data.get('Drought_Tolerance', 'N/A')} (1=Tolerant, 0=Sensitive)"
        elif "Yield_per_plant" in condition:
            return f"Current yield: {data.get('Yield_per_plant', 'N/A')} g/plant (Threshold: >35g)"
        elif "Rainfall_mm" in condition:
            return f"Rainfall adaptation: {data.get('Rainfall_mm', 'N/A')} mm (Optimal for <1000mm)"
        elif "Grain_weight" in condition:
            return f"Grain weight: {data.get('Grain_weight', 'N/A')} g (High quality: >28g)"
        elif "Temperature_C" in condition:
            return f"Temperature adaptation: {data.get('Temperature_C', 'N/A')}°C (Heat tolerant: >25°C)"
        elif "Height" in condition:
            return (
                f"Plant height: {data.get('Height', 'N/A')} cm (Tall variety: >100cm)"
            )
        elif "Heterozygosity" in condition:
            return f"Genetic diversity: {data.get('Heterozygosity', 'N/A')} (High diversity: >0.2)"

        return "Based on phenotypic performance and environmental adaptation analysis"

    def _extract_supporting_data(self, data: Dict, rule: Dict) -> Dict:
        """Extract relevant data that supports the recommendation"""
        supporting_data = {}
        condition = rule["condition"]

        # Extract key metrics mentioned in condition
        metrics = [
            "Yield_per_plant",
            "Height",
            "Grain_weight",
            "Drought_Tolerance",
            "Rainfall_mm",
            "Temperature_C",
            "Heterozygosity",
        ]

        for metric in metrics:
            if metric in condition and metric in data:
                supporting_data[metric] = data[metric]

        return supporting_data

    def get_ontology_graph(self) -> Dict:
        """Generate graph data for 3D visualization"""
        nodes = []
        edges = []

        # Add trait category nodes
        for trait_category, subtraits in self.ontology["traits"].items():
            nodes.append(
                {
                    "id": trait_category,
                    "name": trait_category.replace("_", " ").title(),
                    "type": "trait_category",
                    "size": 25,
                    "color": "#FF6B6B",
                }
            )

            # Add subtrait nodes
            for subtrait in subtraits:
                nodes.append(
                    {
                        "id": subtrait,
                        "name": subtrait.replace("_", " ").title(),
                        "type": "trait",
                        "size": 18,
                        "color": "#4ECDC4",
                    }
                )
                edges.append(
                    {
                        "source": trait_category,
                        "target": subtrait,
                        "type": "contains",
                        "value": 5,
                    }
                )

        # Add gene function nodes
        for gene, function in self.ontology["gene_functions"].items():
            nodes.append(
                {
                    "id": gene,
                    "name": f"{gene}\n{function}",
                    "type": "gene",
                    "size": 15,
                    "color": "#45B7D1",
                }
            )

        # Add breeding strategy nodes
        for strategy, description in self.ontology["breeding_strategies"].items():
            strategy_id = f"strategy_{strategy}"
            nodes.append(
                {
                    "id": strategy_id,
                    "name": f"{strategy.title()}\nStrategy",
                    "type": "strategy",
                    "size": 20,
                    "color": "#96CEB4",
                }
            )

        # Connect genes to traits they influence
        gene_trait_links = {
            "OsSNP001": "Drought_Tolerance",
            "OsSNP002": "Yield_per_plant",
            "OsSNP003": "Height",
            "OsSNP004": "Grain_weight",
            "OsSNP005": "Drought_Tolerance",
            "OsSNP006": "Yield_per_plant",
            "OsSNP007": "Grain_weight",
            "OsSNP008": "Drought_Tolerance",
            "OsSNP009": "Height",
            "OsSNP010": "Yield_per_plant",
        }

        for gene, trait in gene_trait_links.items():
            if any(node["id"] == trait for node in nodes):
                edges.append(
                    {"source": gene, "target": trait, "type": "influences", "value": 8}
                )

        # Connect strategies to trait categories
        strategy_links = {
            "strategy_pyramiding": "yield_related",
            "strategy_backcrossing": "stress_tolerance",
            "strategy_heterosis": "yield_related",
            "strategy_marker_assisted": "genetic_diversity",
        }

        for strategy, trait_cat in strategy_links.items():
            edges.append(
                {
                    "source": strategy,
                    "target": trait_cat,
                    "type": "optimizes",
                    "value": 6,
                }
            )

        return {"nodes": nodes, "links": edges}

    def get_knowledge_summary(self):
        """Get summary of knowledge base contents"""
        return {
            "total_rules": sum(len(rules) for rules in self.rules.values()),
            "rule_categories": list(self.rules.keys()),
            "ontology_entities": {
                "traits": len(self.ontology["traits"]),
                "genes": len(self.ontology["gene_functions"]),
                "strategies": len(self.ontology["breeding_strategies"]),
            },
        }
