pip install -r requirements.txt

streamlit run app.py

python -m venv venv

.\venv\Scripts\activate

python -m streamlit run ui\app.py


npm run dev
python backend.py

netstat -ano | findstr :8000
taskkill /F /PID PID_NUMBER

Based on the interface I built for you, here is a breakdown of what each "screen" and visualization represents in the AI Knowledge-Based System:

1. The Overview Dashboard (Screenshot 1)
This is the "Pulse" of your AI. It shows the scale of the pre-programmed intelligence in the system:

Expert Rules (8): The number of biological "if-then" conditions the AI uses to make decisions (e.g., if yield is high and heat is high, recommend drought-tolerant crosses).
Trait Categories (5): How the AI groups data (Yield, Stress, Growth, Genetics, Environment).
Gene Functions (10): The specific genetic markers (SNPs) the AI "understands" and can link to physical traits.
2. 🧬 Genotype Analysis (Screenshot 2)
This screen is where the AI performs a "Deep Dive" into a specific rice variety:

Knowledge Ontology (Left): A 3D map showing how the selected variety connects to its genetic markers (Genes) and physical characteristics (Traits).
Rule Inference Network (Right): This is unique. It shows the AI's "Train of Thought." It visualizes which specific expert rules were triggered by this variety's data. If the AI recommends a variety for breeding, it shows the "path" it took to reach that conclusion.
3. 🌍 Trait Explorer (Screenshot 3)
This screen is for Pattern Recognition across the entire dataset:

3D Trait Correlation: You can select any three traits (like Yield, Height, and Grain Weight).
What it represents: Each dot is a different rice variety. Its position in 3D space shows its performance across those three metrics.
Color-Coding: The blue-to-white color gradient represents Drought Tolerance. This allows you to visually identify "sweet spots"—clusters of varieties that have both high yield and high drought tolerance.
4. 📊 Knowledge Graphs (Screenshot 4)
This is the Structural Map of the entire project's intelligence:

Complete Knowledge Ontology: It visualizes the entire "Brain" of the project. It shows how high-level Breeding Strategies (like "Heterosis" or "Pyramiding") are linked to specific Trait Categories, which are in turn linked to specific SNP markers.
Use Case: It helps researchers understand the "Big Picture" of how genetic markers influence environmental adaptation strategies.
In summary:

Tab 1 is for analyzing one variety.
Tab 2 is for finding trends in the whole dataset.
Tab 3 is for understanding the AI's internal logic structure.


To help you understand the "brain" of this project, here are the four key concepts that make it an Intelligent Bio-System:

1. How the Search Works (RAG - Retrieval-Augmented Generation)
Unlike a normal search (which just looks for keywords), the BioRetriever uses PlantBERT.

Vector Search: It converts the entire description of a rice variety—its SNPs, its yield, and its environment—into a long string of numbers (a "vector").
Contextual Understanding: This allows you to search for things like "High yield variety for hot climates" even if those exact words aren't in the dataset. The AI finds varieties that are "mathematically close" to that description.
2. How the AI Explains Itself (SHAP)
The SHAP Waterfall Plot you see in the UI is the "X-Ray" of the AI's prediction.

Base Value: The average yield of all varieties in the dataset.
Contributors: The AI looks at the specific SNPs of a variety. If OsSNP001 is "2" and that usually results in better water retention, SHAP shows a Positive (Red) contribution for that SNP.
Scientific Validation: This ensures the AI isn't just "guessing" but is actually weighting specific genetic traits.
3. The Expert Inference Engine
The Rule Inference Network tab shows the "Traditional Intelligence" part of the system.

Knowledge Base: We've programmed human "expert rules" (like: If Rainfall < 1000mm, then it's a drought-prone zone).
Logic Path: While the XGBoost model is a "Black Box" (it learns from data), the Inference Engine is a "White Box"—it follows clear, logical rules that a biologist would use, providing a second layer of verification.
4. Smart Breeding Recommendations
When you ask for a recommendation, the Enhanced Recommendation Engine performs a "Digital Cross":

Complementarity: It looks for Parent A (strong in yield) and Parent B (strong in drought resistance).
Hybrid Simulation: It mathematically averages their genetic markers and then uses the machine learning model to predict the yield of their unborn offspring.
Optimization: It then uses the Expert Rules to find the breeding strategy (like "Pyramiding") that best fits the goal.