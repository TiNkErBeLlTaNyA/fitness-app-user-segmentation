```markdown
# Fitness App User Segmentation

Project: Fitness App User Segmentation  
Purpose: Segment fitness app users using K-Means clustering to help product and marketing teams target users with tailored experiences and retention strategies.

Why this project is valuable
- Demonstrates end-to-end data science workflow for a consumer product (fitness app).
- Shows data simulation, EDA, outlier & missing-value handling, feature engineering, clustering, cluster interpretation, and an interactive Streamlit dashboard for product stakeholders.
- Has unique touches: interactive cluster-driven recommendations, 3D cluster visualizations with Plotly, parallel coordinates to inspect cluster behavior, and a small rule-based "next actions" recommender per cluster.

Contents
- fitness_segmentation.ipynb — Jupyter Notebook (data generation, EDA, clustering, visualization, interpretation)
- app.py — Streamlit dashboard (upload dataset, interactive EDA, cluster visualization and filters)
- generate_dataset.py — script to generate `fitness_users.csv` (500 rows). The notebook will also generate it automatically if not found.
- fitness_users.csv — generated dataset (the notebook and generator create this file if absent)
- requirements.txt — Python dependencies
- README.md — this file

Getting started (quick)
1. Clone the repo.
2. Create a virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .venv\Scripts\activate       # Windows
   pip install -r requirements.txt
3. Generate dataset (if you want to run generator script):
   python generate_dataset.py
   This creates `fitness_users.csv` with 500 simulated users.
4. Open the notebook:
   jupyter lab   # or jupyter notebook
   Run `fitness_segmentation.ipynb`. The notebook will auto-generate `fitness_users.csv` if it's missing.
5. Run the Streamlit dashboard:
   streamlit run app.py
   The dashboard accepts file upload; if none provided it loads `fitness_users.csv`.

What the notebook does (high level)
- Generates a realistic, diverse synthetic dataset of 500 fitness app users.
- Performs EDA (distributions, correlations, pairplots, missing-data patterns).
- Handles missing values (median imputation) and caps outliers using IQR winsorization.
- Scales features and finds good K using elbow + silhouette heuristics, then fits K-Means for 3-5 cluster choices.
- Visualizes clusters using 2D PCA scatter, 3D Plotly scatter, pairplots, and parallel coordinates.
- Profiles clusters and provides business-oriented interpretations and recommended actions per cluster.
- Saves clustered dataset `fitness_users_clustered.csv` for downstream use.

Unique touches (>=30% different than typical tutorials)
- 3D interactive Plotly cluster visualization (rotatable in notebook and Streamlit).
- Parallel coordinates per-cluster to show multi-feature tradeoffs.
- Cluster-driven short recommender: for each cluster the notebook produces suggested product actions (e.g., "Offer beginner strength plans" or "Promote group challenges").
- Streamlit dashboard with interactive filters (age range, gender, workout type) and a mini "what-if" where you can input a sample user's features to see which cluster they'd belong to and recommended next actions.

Tech stack
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn, plotly)
- Streamlit for dashboard
- Jupyter Notebook for analysis and reporting

```
