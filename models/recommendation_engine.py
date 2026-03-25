# models/recommendation_engine.py
import pandas as pd


class RecommendationEngine:
    def __init__(self, retriever, predictor):
        """
        retriever: BioRetriever object with .df
        predictor: PhenotypePredictor object with .predict()
        """
        self.retriever = retriever
        self.predictor = predictor

    # ------------------- GENOTYPE → PHENOTYPE -------------------
    def summarize_genotype(self, genotype_name):
        df = self.retriever.df
        genotype_col = "Genotype" if "Genotype" in df.columns else "Variety"
        row = df[df[genotype_col] == genotype_name]
        if row.empty:
            return f"⚠️ Genotype '{genotype_name}' not found."

        row = row.iloc[0]

        # Extract SNPs
        snp_cols = [c for c in df.columns if c.startswith("OsSNP")]
        snp_summary = ", ".join(f"{c}={row[c]}" for c in snp_cols)

        # Predict phenotype
        snp_values = row[snp_cols]
        pred = self.predictor.predict(snp_values)

        # Include some traits if exist
        extra_traits = []
        for t in ["Yield_per_plant", "Grain_weight", "Drought_Tolerance"]:
            if t in df.columns:
                extra_traits.append(f"{t}={row[t]}")

        extra_summary = ", ".join(extra_traits)

        return (
            f"✅ {genotype_name} SNP pattern: [{snp_summary}].\n"
            f"Predicted Phenotype: {pred}. {extra_summary}.\n"
            f"Suitable as parent for breeding targeting {pred} traits."
        )

    # ------------------- PHENOTYPE → CROSS RECOMMENDATION -------------------
    def recommend_cross(self, trait_text, top_n=2):
        df = self.retriever.df.copy()
        trait_text = trait_text.lower()
        trait_cols = {}

        # Map keywords to dataset columns
        if "yield" in trait_text:
            trait_cols["Yield_per_plant"] = True
        if "grain" in trait_text or "weight" in trait_text:
            trait_cols["Grain_weight"] = True
        if "drought" in trait_text:
            trait_cols["Drought_Tolerance"] = True

        if not trait_cols:
            return "⚠️ No matching traits found in dataset."

        # Normalize trait values for scoring
        for col in trait_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col + "_norm"] = (df[col] - min_val) / (max_val - min_val + 1e-6)

        # Compute combined score
        df["score"] = df[[c + "_norm" for c in trait_cols]].sum(axis=1)

        # Select top 2 parents
        top_parents = df.sort_values("score", ascending=False).head(top_n)
        if len(top_parents) < 2:
            return "⚠️ Not enough varieties for cross recommendation."

        parent1 = top_parents.iloc[0]
        parent2 = top_parents.iloc[1]

        # Combine SNPs
        snp_cols = [c for c in df.columns if c.startswith("OsSNP")]
        snps1 = ", ".join(f"{c}={parent1[c]}" for c in snp_cols)
        snps2 = ", ".join(f"{c}={parent2[c]}" for c in snp_cols)

        # Estimate hybrid traits (average)
        hybrid_traits = {}
        for col in trait_cols:
            hybrid_traits[col] = round((parent1[col] + parent2[col]) / 2, 2)

        return (
            f"✅ Best cross: {parent1['Variety']} × {parent2['Variety']}\n"
            f"- Parent1 SNPs: [{snps1}]\n"
            f"- Parent2 SNPs: [{snps2}]\n"
            f"Expected hybrid traits: "
            + ", ".join(f"{k} ~ {v}" for k, v in hybrid_traits.items())
        )
