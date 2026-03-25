import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class EnhancedRecommendationEngine:
    def __init__(self, retriever, predictor):
        """
        Enhanced engine with knowledge-based reasoning and 3D visualization
        """
        self.retriever = retriever
        self.predictor = predictor

        # Import knowledge components
        from models.knowledge_base import PlantBreedingKnowledgeBase
        from models.knowledge_visualizer_3d import Knowledge3DVisualizer

        self.knowledge_base = PlantBreedingKnowledgeBase()
        self.visualizer = Knowledge3DVisualizer(self.knowledge_base)

        logger.info("✅ Enhanced Recommendation Engine initialized")

    def get_intelligent_recommendations(
        self, genotype_name: str
    ) -> Tuple[str, Any, Any]:
        """Enhanced recommendations with knowledge-based reasoning and SHAP explainability"""
        try:
            df = self.retriever.df
            genotype_col = "Genotype" if "Genotype" in df.columns else "Variety"
            row = df[df[genotype_col] == genotype_name]

            if row.empty:
                return "⚠️ Genotype not found in database.", None, None

            row_data = row.iloc[0].to_dict()

            # Get knowledge-based recommendations
            kb_recommendations = self.knowledge_base.infer_recommendations(row_data)

            # Get SHAP insights
            shap_values, base_value = self.predictor.get_shap_explanation(row_data)
            shap_insights = self._get_top_shap_insights(shap_values)

            # Generate 3D visualizations
            knowledge_graph = self.visualizer.create_3d_knowledge_graph()
            rule_network = self.visualizer.create_3d_rule_network(kb_recommendations)

            # Format comprehensive output
            recommendation_text = self._format_recommendations(
                genotype_name, row_data, kb_recommendations, shap_insights
            )

            logger.info(f"✅ Generated intelligent recommendations for {genotype_name}")
            return recommendation_text, knowledge_graph, rule_network

        except Exception as e:
            logger.error(f"Error in intelligent recommendations: {e}")
            return f"❌ Error generating recommendations: {str(e)}", None, None

    def _get_top_shap_insights(self, shap_values) -> List[Dict]:
        """Extract top positive and negative drivers from SHAP values"""
        if shap_values is None:
            return []
        
        insights = []
        feature_names = self.predictor.feature_names
        
        # Create list of (feature, value)
        contributions = []
        for i, val in enumerate(shap_values):
            if i < len(feature_names):
                contributions.append({"feature": feature_names[i], "value": val})
        
        # Sort by absolute value
        contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return contributions[:5] # Top 5 drivers

    def _format_recommendations(
        self, genotype_name: str, row_data: Dict, recommendations: List[Dict], shap_insights: List[Dict] = []
    ) -> str:
        """Format recommendations into readable text including SHAP insights"""
        output = f"🧠 **AI-Powered Analysis for {genotype_name}**\n\n"

        # Explainability Drivers
        if shap_insights:
            output += "🧪 **Top Performance Drivers (SHAP Analysis):**\n"
            for insight in shap_insights:
                feature = insight["feature"].replace("_", " ").title()
                impact = "📈 Booster" if insight["value"] > 0 else "📉 Friction"
                output += f"   • **{feature}**: {impact} ({insight['value']:+.2f} yield impact)\n"
            output += "\n"

        # Basic information
        output += "📊 **Current Traits:**\n"
        basic_info = [
            ("Yield_per_plant", "Yield (g/plant)"),
            ("Height", "Height (cm)"),
            ("Grain_weight", "Grain Weight (g)"),
            ("Drought_Tolerance", "Drought Tolerance"),
            ("Rainfall_mm", "Rainfall (mm)"),
        ]

        for key, display in basic_info:
            if key in row_data and pd.notna(row_data[key]):
                value = row_data[key]
                if key == "Drought_Tolerance":
                    value = "Tolerant" if value == 1 else "Sensitive"
                output += f"   • {display}: {value}\n"

        output += "\n🎯 **AI Recommendations:**\n"

        if not recommendations:
            output += "   No specific recommendations based on current data.\n"
        else:
            for i, rec in enumerate(recommendations[:5], 1):
                output += f"{i}. **{rec['recommendation']}**\n"
                output += f"   📈 Confidence: {rec['confidence']:.0%}\n"
                output += f"   💡 Reasoning: {rec['reasoning']}\n"
                output += f"   🎯 Category: {rec['category']}\n\n"

        # Breeding potential
        output += self._assess_breeding_potential(row_data)

        return output

    def _assess_breeding_potential(self, row_data: Dict) -> str:
        """Assess breeding potential based on genetic and phenotypic data"""
        output = "🔬 **Breeding Potential Assessment:**\n"

        scores = []

        # Yield potential
        if row_data.get("Yield_per_plant", 0) > 35:
            scores.append(("High Yield Potential", "✅"))
        elif row_data.get("Yield_per_plant", 0) > 25:
            scores.append(("Medium Yield Potential", "⚠️"))
        else:
            scores.append(("Low Yield Potential", "❌"))

        # Drought tolerance
        if row_data.get("Drought_Tolerance") == 1:
            scores.append(("Drought Tolerant", "✅"))
        else:
            scores.append(("Drought Sensitive", "❌"))

        # Genetic diversity
        if row_data.get("Heterozygosity", 0) > 0.2:
            scores.append(("High Genetic Diversity", "✅"))
        else:
            scores.append(("Low Genetic Diversity", "⚠️"))

        for trait, status in scores:
            output += f"   • {status} {trait}\n"

        # Overall assessment
        positive_scores = sum(1 for _, status in scores if status == "✅")
        if positive_scores >= 2:
            output += "\n   🏆 **Overall: Excellent breeding parent**\n"
        elif positive_scores >= 1:
            output += "\n   ⚠️ **Overall: Moderate breeding potential**\n"
        else:
            output += "\n   🔴 **Overall: Limited breeding value**\n"

        return output

    def recommend_optimized_cross(
        self, breeding_goal: str, top_n: int = 3
    ) -> Tuple[str, Any]:
        """Enhanced cross recommendation with knowledge reasoning"""
        try:
            df = self.retriever.df.copy()
            breeding_goal = breeding_goal.lower()

            # Map breeding goals to traits
            trait_mapping = {
                "yield": "Yield_per_plant",
                "height": "Height",
                "grain": "Grain_weight",
                "drought": "Drought_Tolerance",
                "quality": "Grain_weight",
            }

            target_traits = []
            for keyword, trait in trait_mapping.items():
                if keyword in breeding_goal and trait in df.columns:
                    target_traits.append(trait)

            if not target_traits:
                return "⚠️ No matching traits found for your breeding goal.", None

            # Score varieties based on target traits
            df["breeding_score"] = 0
            for trait in target_traits:
                if trait in df.columns:
                    # Normalize and weight the trait
                    min_val = df[trait].min()
                    max_val = df[trait].max()
                    if max_val > min_val:
                        df[trait + "_norm"] = (df[trait] - min_val) / (
                            max_val - min_val
                        )
                        df["breeding_score"] += df[trait + "_norm"]

            # Select top parents
            top_parents = df.nlargest(top_n * 2, "breeding_score")

            if len(top_parents) < 2:
                return "⚠️ Not enough varieties for cross recommendation.", None

            # Generate cross recommendations
            recommendations = []
            for i in range(min(3, len(top_parents) - 1)):
                parent1 = top_parents.iloc[i]
                parent2 = top_parents.iloc[i + 1]

                cross_info = self._analyze_cross(parent1, parent2, target_traits)
                recommendations.append(cross_info)

            # Format output
            output = self._format_cross_recommendations(breeding_goal, recommendations)

            # Create knowledge graph
            knowledge_graph = self.visualizer.create_3d_knowledge_graph()

            logger.info("✅ Generated optimized cross recommendations")
            return output, knowledge_graph

        except Exception as e:
            logger.error(f"Error in optimized cross recommendation: {e}")
            return f"❌ Error generating cross recommendations: {str(e)}", None

    def _analyze_cross(self, parent1, parent2, target_traits: List[str]) -> Dict:
        """Analyze a specific cross between two parents"""
        # Calculate expected hybrid values (average)
        hybrid_traits = {}
        for trait in target_traits:
            if trait in parent1 and trait in parent2:
                hybrid_traits[trait] = (parent1[trait] + parent2[trait]) / 2

        # Get SNP information
        snp_cols = [col for col in parent1.index if col.startswith("OsSNP")]
        parent1_snps = {col: parent1[col] for col in snp_cols if pd.notna(parent1[col])}
        parent2_snps = {col: parent2[col] for col in snp_cols if pd.notna(parent2[col])}

        return {
            "parent1_name": parent1.get("Variety", "Unknown"),
            "parent2_name": parent2.get("Variety", "Unknown"),
            "parent1_snps": parent1_snps,
            "parent2_snps": parent2_snps,
            "hybrid_traits": hybrid_traits,
            "complementarity_score": self._calculate_complementarity(
                parent1, parent2, target_traits
            ),
        }

    def _calculate_complementarity(
        self, parent1, parent2, target_traits: List[str]
    ) -> float:
        """Calculate how complementary two parents are for target traits"""
        score = 0
        for trait in target_traits:
            if trait in parent1 and trait in parent2:
                # Higher score if one parent is strong where the other is weak
                trait_range = max(parent1[trait], parent2[trait]) - min(
                    parent1[trait], parent2[trait]
                )
                score += trait_range
        return score / len(target_traits) if target_traits else 0

    def _format_cross_recommendations(
        self, breeding_goal: str, recommendations: List[Dict]
    ) -> str:
        """Format cross recommendations into readable text"""
        output = (
            f"🎯 **Smart Cross Recommendations for: '{breeding_goal.title()}'**\n\n"
        )

        for i, rec in enumerate(recommendations, 1):
            output += f"**Cross {i}: {rec['parent1_name']} × {rec['parent2_name']}**\n"
            output += (
                f"   📊 Complementarity Score: {rec['complementarity_score']:.2f}\n"
            )

            # Expected hybrid performance
            output += "   🧬 Expected Hybrid Traits:\n"
            for trait, value in rec["hybrid_traits"].items():
                trait_name = trait.replace("_", " ").title()
                output += f"      • {trait_name}: {value:.2f}\n"

            # SNP information (first 3 SNPs as example)
            output += "   🧪 Key SNP Patterns:\n"
            snp_count = 0
            for snp, value in list(rec["parent1_snps"].items())[:3]:
                output += f"      • {snp}: Parent1={value}, Parent2={rec['parent2_snps'].get(snp, 'N/A')}\n"
                snp_count += 1

            output += "\n"

        # Add knowledge-based strategy
        kb_strategies = self.knowledge_base.ontology["breeding_strategies"]
        strategy = list(kb_strategies.values())[0]  # Get first strategy

        output += f"💡 **Knowledge-Based Strategy:** {strategy}\n"
        output += "🔍 *Tip: Consider environmental adaptation and local growing conditions when selecting parents.*"

        return output

    def get_knowledge_summary(self) -> Dict:
        """Get summary of the knowledge system"""
        return self.knowledge_base.get_knowledge_summary()
