from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import pandas as pd
import numpy as np
import plotly.io as pio
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retriever_bio import BioRetriever
from models.predictor_xgboost import PhenotypePredictor
from models.enhanced_recommendation_engine import EnhancedRecommendationEngine
from utils.config import DATASET_PATH

app = FastAPI(title="Agri-Tech API")

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (Lazy Loaded)
retriever = None
predictor = None
enhanced_engine = None

def load_models():
    global retriever, predictor, enhanced_engine
    if retriever is None:
        try:
            logger.info("⏳ Loading models (lazy-load)...")
            retriever = BioRetriever(DATASET_PATH)
            predictor = PhenotypePredictor(DATASET_PATH)
            predictor.train()
            enhanced_engine = EnhancedRecommendationEngine(retriever, predictor)
            logger.info("✅ Models initialized successfully")
        except Exception as e:
            logger.error(f"❌ Lazy-load initialization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")

@app.get("/api/init")
async def init_data():
    """Returns basic configuration and available genotypes"""
    load_models()
    genotype_col = "Variety" if "Variety" in retriever.df.columns else "Genotype"
    return {
        "genotypes": retriever.df[genotype_col].unique().tolist(),
        "traits": [c for c in retriever.df.columns if c not in ["SampleID", genotype_col, "Country", "Group", "State", "Soil_Type"]],
        "states": retriever.df["State"].dropna().unique().tolist()
    }

@app.get("/api/search")
async def search_genotypes(query: str):
    """Search for genotypes based on traits, varieties, or locations.
    Supports semantic phrases like 'high yield', 'drought tolerant', 'tall plant', etc."""
    load_models()
    if not query or len(query) < 2:
        return []

    df = retriever.df.copy().fillna("")
    q = query.strip().lower()

    # ── Semantic keyword → numeric filter mappings ──────────────────────────
    TRAIT_RULES = [
        # Yield
        (["high yield", "max yield", "best yield", "high production"],
         lambda d: d[pd.to_numeric(d["Yield_per_plant"], errors="coerce") >= 35]),
        (["low yield"],
         lambda d: d[pd.to_numeric(d["Yield_per_plant"], errors="coerce") < 30]),
        # Height
        (["tall", "tall plant", "high height", "max height"],
         lambda d: d[pd.to_numeric(d["Height"], errors="coerce") >= 110]),
        (["short", "dwarf", "short plant"],
         lambda d: d[pd.to_numeric(d["Height"], errors="coerce") <= 85]),
        # Grain weight / protein proxy
        (["high protein", "high grain", "heavy grain", "good quality grain", "protein"],
         lambda d: d[pd.to_numeric(d["Grain_weight"], errors="coerce") >= 28]),
        (["light grain", "low grain weight"],
         lambda d: d[pd.to_numeric(d["Grain_weight"], errors="coerce") < 22]),
        # Drought
        (["drought tolerant", "drought tolerance", "drought resistant", "dry condition"],
         lambda d: d[d["Drought_Tolerance"].astype(str).str.lower().isin(["1", "tolerant", "yes"])]),
        (["drought sensitive", "water intensive", "needs water"],
         lambda d: d[d["Drought_Tolerance"].astype(str).str.lower().isin(["0", "sensitive", "no"])]),
        # Rainfall
        (["low rainfall", "less water", "arid"],
         lambda d: d[pd.to_numeric(d["Rainfall_mm"], errors="coerce") <= 900]),
        (["high rainfall", "more water", "wet", "flood"],
         lambda d: d[pd.to_numeric(d["Rainfall_mm"], errors="coerce") >= 1500]),
        # Temperature
        (["heat tolerant", "warm climate", "hot region"],
         lambda d: d[pd.to_numeric(d["Temperature_C"], errors="coerce") >= 28]),
        (["cool climate", "cold tolerant", "low temperature"],
         lambda d: d[pd.to_numeric(d["Temperature_C"], errors="coerce") <= 22]),
        # Soil
        (["clay soil", "clay"],
         lambda d: d[d["Soil_Type"].astype(str).str.lower().str.contains("clay")]),
        (["sandy soil", "sandy"],
         lambda d: d[d["Soil_Type"].astype(str).str.lower().str.contains("sandy|sand")]),
        (["loam", "loamy"],
         lambda d: d[d["Soil_Type"].astype(str).str.lower().str.contains("loam")]),
        (["laterite"],
         lambda d: d[d["Soil_Type"].astype(str).str.lower().str.contains("laterite")]),
        # Best overall
        (["best variety", "top variety", "recommended", "all rounder"],
         lambda d: d.sort_values("Yield_per_plant", ascending=False)),
    ]

    # Try semantic match first
    matched = pd.DataFrame()
    found_semantic = False
    for keywords, filterfn in TRAIT_RULES:
        if any(kw in q for kw in keywords):
            try:
                matched = filterfn(df.copy())
                found_semantic = True
            except Exception:
                matched = pd.DataFrame()
            break

    # If it's a generic trait name, fallback to returning top matches
    if not found_semantic or len(matched) == 0:
        if "trait" in q or "phenotype" in q or "genotype" in q or "variety" in q:
            matched = df.copy()
        elif "yield" in q:
            matched = df.copy()
        elif "height" in q:
            matched = df.copy()
        elif "grain" in q or "weight" in q or "protein" in q:
            matched = df[pd.to_numeric(df["Grain_weight"], errors="coerce") >= 25]
        elif "drought" in q or "tolerance" in q or "tolerant" in q:
            matched = df[df["Drought_Tolerance"].astype(str).str.lower().isin(["1", "tolerant", "yes"])]
        elif "rain" in q:
            matched = df.copy()
        elif "temp" in q or "climate" in q:
            matched = df.copy()
        elif "soil" in q:
            matched = df[df["Soil_Type"].astype(str).str.strip() != ""]

    # Fall back to full-text search across all columns
    if len(matched) == 0:
        mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
        matched = df[mask]

    # Sort appropriately, default to yield descending
    try:
        matched = matched.copy()
        if "height" in q and not any(kw in q for kw in ["short", "dwarf"]):
            matched["_sort_num"] = pd.to_numeric(matched["Height"], errors="coerce")
        elif "rain" in q and "low" not in q:
            matched["_sort_num"] = pd.to_numeric(matched["Rainfall_mm"], errors="coerce")
        elif ("temp" in q or "climate" in q) and "cool" not in q and "cold" not in q:
            matched["_sort_num"] = pd.to_numeric(matched["Temperature_C"], errors="coerce")
        elif "grain" in q or "weight" in q or "protein" in q:
            matched["_sort_num"] = pd.to_numeric(matched["Grain_weight"], errors="coerce")
        else:
            matched["_sort_num"] = pd.to_numeric(matched["Yield_per_plant"], errors="coerce")
            
        matched = matched.sort_values("_sort_num", ascending=False).drop(columns=["_sort_num"])
    except Exception:
        pass

    results = matched.head(20)
    logger.info(f"Search '{query}' → {len(results)} results")
    return results.to_dict(orient="records")


@app.get("/api/genotype/{name}")
async def get_genotype_details(name: str):
    """Get detailed traits for a specific variety"""
    load_models()
    genotype_col = "Variety" if "Variety" in retriever.df.columns else "Genotype"
    row = retriever.df[retriever.df[genotype_col].astype(str).str.lower() == name.lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail="Genotype not found")
    
    # Get deep analysis
    recs, kg_fig, rule_fig = enhanced_engine.get_intelligent_recommendations(row.iloc[0][genotype_col])
    
    return {
        "data": row.iloc[0].to_dict(),
        "recommendations": recs,
        "kg_graph": json.loads(pio.to_json(kg_fig)) if kg_fig else None,
        "rule_graph": json.loads(pio.to_json(rule_fig)) if rule_fig else None
    }

@app.get("/api/recommend")
async def get_recommendations(goal: str):
    """AI breeding recommendations"""
    load_models()
    rec_text, kg_fig = enhanced_engine.recommend_optimized_cross(goal)
    return {
        "text": rec_text,
        "kg_graph": json.loads(pio.to_json(kg_fig)) if kg_fig else None
    }

@app.get("/api/kg")
async def get_global_kg():
    """Return the global 3D Knowledge Graph"""
    load_models()
    try:
        fig = enhanced_engine.visualizer.create_3d_knowledge_graph(width=1000, height=600)
        return json.loads(pio.to_json(fig))
    except Exception as e:
        logger.error(f"Error in /api/kg: {e}")
        return {"error": str(e)}

@app.get("/api/traits")
async def get_trait_correlation(t1: str = "Yield_per_plant", t2: str = "Height", t3: str = "Grain_weight"):
    """Return 3D Trait Correlation plot"""
    load_models()
    try:
        fig = enhanced_engine.visualizer.create_trait_correlation_3d(retriever.df, traits=[t1, t2, t3], width=1000, height=600)
        return json.loads(pio.to_json(fig))
    except Exception as e:
        logger.error(f"Error in /api/traits: {e}")
        return {"error": str(e)}

@app.get("/api/map")
async def get_map_data():
    """Return state-level variety distribution as card data"""
    load_models()
    states = retriever.df["State"].dropna().unique()
    map_data = []
    for s in states:
        count = len(retriever.df[retriever.df["State"] == s])
        map_data.append({"state": s, "count": count})
    return map_data

@app.get("/api/mapfig")
async def get_india_map_figure(variety: str = None):
    """Return a Plotly ScatterGeo figure. Optional ?variety=NAME filters by cultivated variety."""
    load_models()
    import plotly.graph_objects as go

    STATE_COORDS = {
        "Andhra Pradesh": (15.9129, 79.7400), "Assam": (26.2006, 92.9376),
        "Bihar": (25.0961, 85.3131), "Chhattisgarh": (21.2787, 81.8661),
        "Gujarat": (22.2587, 71.1924), "Haryana": (29.0588, 76.0856),
        "Himachal Pradesh": (31.1048, 77.1734), "Jharkhand": (23.6102, 85.2799),
        "Karnataka": (15.3173, 75.7139), "Kerala": (10.8505, 76.2711),
        "Madhya Pradesh": (22.9734, 78.6569), "Maharashtra": (19.7515, 75.7139),
        "Manipur": (24.6637, 93.9063), "Odisha": (20.9517, 85.0985),
        "Punjab": (31.1471, 75.3412), "Rajasthan": (27.0238, 74.2179),
        "Tamil Nadu": (11.1271, 78.6569), "Telangana": (18.1124, 79.0193),
        "Uttar Pradesh": (26.8467, 80.9462), "Uttarakhand": (30.0668, 79.0193),
        "West Bengal": (22.9868, 87.8550), "Goa": (15.2993, 74.1240),
        "Tripura": (23.9408, 91.9882), "Meghalaya": (25.4670, 91.3662),
        "Nagaland": (26.1584, 94.5624), "Mizoram": (23.1645, 92.9376),
        "Arunachal Pradesh": (28.2180, 94.7278), "Sikkim": (27.5330, 88.5122),
    }

    # Case-insensitive lookup; filter by variety if provided
    df = retriever.df.copy()
    genotype_col = "Variety" if "Variety" in df.columns else "Genotype"
    df["State"] = df["State"].astype(str).str.strip().str.title()

    if variety:
        df_filtered = df[df[genotype_col].astype(str).str.lower() == variety.lower()]
    else:
        df_filtered = df

    lats, lons, counts, texts = [], [], [], []
    for state, (lat, lon) in STATE_COORDS.items():
        count = len(df_filtered[df_filtered["State"].str.lower() == state.lower()])
        if count > 0:
            lats.append(lat)
            lons.append(lon)
            counts.append(count)
            label = (f"<b>{state}</b><br>{count} occurrence(s) of {variety}"
                     if variety else f"<b>{state}</b><br>{count} Plant Varieties")
            texts.append(label)

    if not counts:
        return {"error": "No data found for this variety"}

    title = (f"🌾 States where {variety} is cultivated" if variety
             else "🗺️ INDIA — Plant Variety Distribution by State")

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons,
        text=texts,
        hoverinfo="text",
        hovertemplate="%{text}<extra></extra>",
        mode="markers",
        marker=dict(
            size=12,
            color="#a8ff78",
            opacity=0.95,
            line=dict(width=2, color="#000"),
            symbol="circle",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, y=0.97,
            font=dict(size=17, color="white", family="Arial Black"),
        ),
        geo=dict(
            scope="asia",
            resolution=50,
            showland=True,       landcolor="#1a2e1a",
            showocean=True,      oceancolor="#04040f",
            showcountries=True,  countrycolor="rgba(255,255,255,0.7)",
            showcoastlines=True, coastlinecolor="rgba(255,255,255,0.5)",
            showsubunits=True,   subunitcolor="rgba(255,255,255,0.3)",  # state borders
            showlakes=False,
            center=dict(lat=22, lon=80),
            projection_scale=4.5,
            bgcolor="rgba(0,0,0,0)",
            lataxis=dict(range=[7, 37]),
            lonaxis=dict(range=[67, 98]),
        ),
        paper_bgcolor="rgba(0,0,0,1)",
        plot_bgcolor="rgba(0,0,0,1)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
        width=900, height=560,
    )

    logger.info(f"✅ India map: variety='{variety}', {len(counts)} states shown")
    return json.loads(pio.to_json(fig))

# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting backend server on port 8000...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
