import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import base64
import pandas as pd
import numpy as np
import logging
from streamlit_shap import st_shap
import shap
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def add_bg_from_local(image_file):
    """Add a local background image to Streamlit UI with enhanced styling"""
    try:
        # If the file doesn't exist, we'll use a premium gradient
        if not os.path.exists(image_file):
            raise FileNotFoundError

        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=Outfit:wght@400;600;800&display=swap');

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: radial-gradient(circle at center, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.8) 100%);
            z-index: -1;
        }}
        .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .main-title {{
            text-align: center;
            font-family: 'Outfit', sans-serif;
            font-size: 52px !important;
            font-weight: 800;
            background: linear-gradient(to right, #ffffff, #a8ff78, #78ffd6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 30px;
            letter-spacing: -1px;
        }}
        /* Professional Glassmorphism Card */
        .card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            margin-top: 20px;
            transition: transform 0.3s ease;
        }}
        
        /* Interactive Metrics */
        div[data-testid="stMetricValue"] {{
            font-size: 36px;
            font-weight: 800;
            color: #a8ff78 !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: #ffffff !important;
            opacity: 0.8;
            font-size: 14px !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Global Text Visibility Fix */
        h1, h2, h3, p, span, label, div {{
            color: #ffffff !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(255,255,255,0.05);
            border-radius: 12px;
            color: white;
            font-weight: 600;
            border: none;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #a8ff78 !important;
            color: #000 !important;
        }}
        
        /* Interactive Radio Chips */
        div[role="radiogroup"] {{
            flex-direction: row !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
            background: transparent !important;
            border: none !important;
            gap: 15px !important;
            width: 100% !important;
        }}
        div[role="radiogroup"] > label {{
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            padding: 12px 24px !important;
            border-radius: 50px !important;
            transition: all 0.3s ease !important;
            margin: 0 !important;
        }}
        div[role="radiogroup"] > label:hover {{
            background: rgba(255,255,255,0.12) !important;
            border-color: #a8ff78 !important;
            transform: translateY(-2px);
        }}
        div[role="radiogroup"] [data-testid="stWidgetSelectionPiece"] {{
            background-color: #a8ff78 !important;
            color: #000 !important;
        }}
        
        /* Premium Text Input */
        .stTextInput > div > div > input {{
            background-color: rgba(255,255,255,0.05) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 14px !important;
            padding: 12px 20px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: #a8ff78 !important;
            background-color: rgba(255,255,255,0.1) !important;
            box-shadow: 0 0 20px rgba(168, 255, 120, 0.2) !important;
        }}
        
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Background image not found: {e}")
        # Fallback background - Premium Mesh Gradient
        st.markdown(
            """
        <style>
        .stApp {
            background-color: #0d1117;
            background-image: 
                radial-gradient(at 0% 0%, hsla(120,100%,15%,0.3) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(210,100%,20%,0.3) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(280,100%,15%,0.3) 0, transparent 50%);
        }
        h1, h2, h3, p, span, label, div {
            color: #ffffff !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )


# ---------------- BACKGROUND ----------------
add_bg_from_local(r"C:\Users\lenovo\.gemini\antigravity\brain\c7bee3fa-6ac6-41c1-bead-779d610a6144\agri_tech_background_1773056753007.png")

# ---------------- STATE COORDINATES ----------------
STATE_COORDS = {
    'Andaman and Nicobar Islands': [11.7401, 92.6586],
    'Andhra Pradesh': [15.9129, 79.7400],
    'Arunachal Pradesh': [28.2180, 94.7278],
    'Assam': [26.2006, 92.9376],
    'Bihar': [25.0961, 85.3131],
    'Chandigarh': [30.7333, 76.7794],
    'Chhattisgarh': [21.2787, 81.8661],
    'Dadra and Nagar Haveli': [20.1809, 73.0169],
    'Delhi': [28.7041, 77.1025],
    'Goa': [15.2993, 74.1240],
    'Gujarat': [22.2587, 71.1924],
    'Haryana': [29.0588, 76.0856],
    'Himachal Pradesh': [31.1048, 77.1665],
    'Jammu and Kashmir': [33.7782, 76.5762],
    'Jharkhand': [23.6102, 85.2799],
    'Karnataka': [15.3173, 75.7139],
    'Kerala': [10.8505, 76.2711],
    'Lakshadweep': [10.5667, 72.6417],
    'Madhya Pradesh': [22.9734, 78.6569],
    'Maharashtra': [19.7515, 75.7139],
    'Manipur': [24.6637, 93.9063],
    'Meghalaya': [25.4670, 91.3662],
    'Mizoram': [23.1645, 92.9376],
    'Nagaland': [26.1584, 94.5624],
    'Odisha': [20.9517, 85.9000],
    'Puducherry': [11.9416, 79.8083],
    'Punjab': [31.1471, 75.3412],
    'Rajasthan': [27.0238, 74.2179],
    'Sikkim': [27.5330, 88.5122],
    'Tamil Nadu': [11.1271, 78.6569],
    'Telangana': [18.1124, 79.0193],
    'Tripura': [23.9408, 91.9882],
    'Uttar Pradesh': [26.8467, 80.9462],
    'Uttarakhand': [30.0668, 79.0193],
    'West Bengal': [22.9868, 87.8550]
}

# ---------------- IMPORT PROJECT MODULES ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models.retriever_bio import BioRetriever
    from models.predictor_xgboost import PhenotypePredictor
    from models.hybrid_simulator import cross_genotypes
    from models.enhanced_recommendation_engine import EnhancedRecommendationEngine
    logger.info("✅ All modules imported successfully")
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.stop()

from utils.config import DATASET_PATH

# ---------------- APP TITLE ----------------
st.markdown(
    '<h1 class="main-title">🌱 Genotype × Phenotype Explorer</h1>',
    unsafe_allow_html=True,
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    """Load models with caching"""
    try:
        retriever = BioRetriever(DATASET_PATH)
        predictor = PhenotypePredictor(DATASET_PATH)
        predictor.train()
        enhanced_engine = EnhancedRecommendationEngine(retriever, predictor)
        logger.info("✅ Models loaded successfully")
        return retriever, predictor, enhanced_engine
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        return None, None, None

retriever, predictor, enhanced_engine = load_models()

if retriever is None:
    st.error("Failed to load models. Please check the data path.")
    st.stop()

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = 0

PAGES = [
    "🔍 Genotype Search",
    "🔬 Phenotype Discovery",
    "🤖 AI Recommendations",
    "🧬 Genotype Analysis",
    "🌍 Trait Explorer",
    "📊 Knowledge Graphs",
    "🗺️ Map Explorer"
]

# ---------------- NAVIGATION ----------------
st.session_state.page = max(0, min(st.session_state.page, len(PAGES) - 1))
nav_cols = st.columns([1, 15, 1])

with nav_cols[0]:
    if st.button("◀", key="top_prev", use_container_width=True):
        if st.session_state.page > 0:
            st.session_state.page -= 1
            st.rerun()

with nav_cols[1]:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <p style="font-size: 14px; opacity: 0.6; margin-bottom: 5px; color: #fff;">STEP {st.session_state.page + 1} OF {len(PAGES)}</p>
        <h2 style="margin-top: 0; color: #a8ff78; letter-spacing: 1px;">{PAGES[st.session_state.page]}</h2>
        <div style="display: flex; justify-content: center; gap: 8px; margin-top: 10px;">
            {' '.join(['<div style="width: 10px; height: 10px; border-radius: 50%; background: ' + ('#a8ff78' if i == st.session_state.page else 'rgba(255,255,255,0.2)') + ';"></div>' for i in range(len(PAGES))])}
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_cols[2]:
    if st.button("▶", key="top_next", use_container_width=True):
        if st.session_state.page < len(PAGES) - 1:
            st.session_state.page += 1
            st.rerun()

genotype_col = "Variety" if "Variety" in retriever.df.columns else "Genotype"

# ---------------- DISPATCHER ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

# PAGE 0: Search
if st.session_state.page == 0:
    st.subheader(f"🧬 {PAGES[0]}")
    trait_input = st.text_input("Search for traits or genotypes (e.g., High Yield):", key="p0_in")
    if trait_input:
        df = retriever.df.copy()
        mask = df.astype(str).apply(lambda x: x.str.contains(trait_input, case=False)).any(axis=1)
        results = df[mask]
        if not results.empty:
            st.dataframe(results[[genotype_col, "Yield_per_plant", "Height"]].head(20))
        else: st.warning("No matches found.")
    else:
        st.info("💡 Tip: Search for 'Tolerant' to find resilient varieties.")
        st.dataframe(retriever.df[[genotype_col, "Yield_per_plant", "Height"]].head(10))

# PAGE 1: Discovery
elif st.session_state.page == 1:
    st.subheader(f"🔬 {PAGES[1]}")
    g_input = st.selectbox("Select Genotype to Explore Traits:", options=retriever.df[genotype_col].unique(), key="p1_sel")
    if g_input:
        r = retriever.df[retriever.df[genotype_col] == g_input].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            for t in ["Yield_per_plant", "Height", "Grain_weight"]:
                st.metric(t.replace("_", " "), r[t])
        with c2:
            st.metric("Climate Adaptation", "Tolerant" if r["Drought_Tolerance"] == 1 else "Sensitive")
            st.write(f"Region: {r['State']}, {r['Country']}")

# PAGE 2: AI Recommendations
elif st.session_state.page == 2:
    st.subheader(f"🤖 {PAGES[2]}")
    goal = st.text_input("What is your breeding target?", placeholder="e.g. increase yield in drought areas", key="p2_in")
    if goal:
        text, kg = enhanced_engine.recommend_optimized_cross(goal)
        st.markdown(text)
        if kg: st.plotly_chart(kg, use_container_width=True)
    else: st.info("Enter a breeding goal to get AI-powered crossing recommendations.")

# PAGE 3: Analysis
elif st.session_state.page == 3:
    st.subheader(f"🧬 {PAGES[3]}")
    variety = st.selectbox("Variety for deep AI analysis:", options=retriever.df[genotype_col].unique(), key="p3_sel")
    if variety:
        recs, kg, net = enhanced_engine.get_intelligent_recommendations(variety)
        st.markdown(recs)
        if kg: st.plotly_chart(kg, use_container_width=True)

# PAGE 4: Trait Explorer
elif st.session_state.page == 4:
    st.subheader(f"🌍 {PAGES[4]}")
    traits = ["Yield_per_plant", "Height", "Grain_weight", "Rainfall_mm", "Temperature_C"]
    sel = st.multiselect("Select 3 traits for 3D analysis:", options=traits, default=traits[:3], key="p4_traits")
    if len(sel) >= 3:
        fig = enhanced_engine.visualizer.create_trait_correlation_3d(retriever.df, traits=sel)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 5: Knowledge Graphs
elif st.session_state.page == 5:
    st.subheader(f"📊 {PAGES[5]}")
    with st.spinner("Loading 3D ontology..."):
        fig = enhanced_engine.visualizer.create_3d_knowledge_graph()
        st.plotly_chart(fig, use_container_width=True)

# PAGE 6: Map Explorer
elif st.session_state.page == 6:
    st.subheader(f"🗺️ {PAGES[6]}")
    states = retriever.df["State"].dropna().unique()
    m_data = []
    LKP = {k.lower(): v for k, v in STATE_COORDS.items()}
    for s in states:
        sc = s.lower().strip()
        if sc in LKP:
            m_data.append({"state": s, "lat": LKP[sc][0], "lon": LKP[sc][1], "count": len(retriever.df[retriever.df["State"]==s])})
    if m_data:
        dfm = pd.DataFrame(m_data)
        fig = px.scatter_mapbox(dfm, lat="lat", lon="lon", size="count", color="count", zoom=3, height=600, mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No geographic data available.")

# FOOTER NAVIGATION
st.markdown("<br><hr style='opacity:0.1;'>", unsafe_allow_html=True)
prev_col, mid_col, next_col = st.columns([1, 4, 1])
with prev_col:
    if st.session_state.page > 0:
        if st.button("◀ Back", key="btn_prev", use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
with next_col:
    if st.session_state.page < len(PAGES) - 1:
        if st.button("Next ▶", key="btn_next", use_container_width=True):
            st.session_state.page += 1
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# FINAL FOOTER
st.markdown(
    """
    <div style='text-align: center; padding: 30px; margin-top: 50px; background: rgba(0,0,0,0.3); border-radius: 20px;'>
        <h3 style='color: #a8ff78;'>🌱 Ag-Tech Intelligence Platform</h3>
        <p style='opacity: 0.6;'>Advanced Agricultural Research Platform | AI-Powered Plant Breeding</p>
    </div>
    """,
    unsafe_allow_html=True
)
