import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Fitness App User Segmentation", layout="wide")

st.title("Fitness App User Segmentation")
st.markdown("Upload a dataset or use the default generated dataset. Explore EDA, clustering and get actionable recommendations per segment.")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("fitness_users.csv")
        except FileNotFoundError:
            st.error("fitness_users.csv not found. Please upload a file or run generate_dataset.py / the notebook to create it.")
            return None
    return df

uploaded = st.sidebar.file_uploader("Upload fitness_users.csv", type=["csv"])
df = load_data(uploaded)

if df is None:
    st.stop()

st.sidebar.header("Filters")
age_min, age_max = st.sidebar.slider("Age range", int(df['Age'].min()), int(df['Age'].max()), (20, 50))
genders = st.sidebar.multiselect("Gender", options=df['Gender'].unique().tolist(), default=df['Gender'].unique().tolist())
workouts = st.sidebar.multiselect("Preferred workout type", options=df['PreferredWorkoutType'].unique().tolist(), default=df['PreferredWorkoutType'].unique().tolist())

filtered = df[(df['Age'] >= age_min) & (df['Age'] <= age_max) & (df['Gender'].isin(genders)) & (df['PreferredWorkoutType'].isin(workouts))]

st.sidebar.header("Clustering")
n_clusters = st.sidebar.slider("Number of clusters (K)", 3, 6, 4)
run_cluster = st.sidebar.button("Run K-Means")

with st.expander("Preview data"):
    st.dataframe(filtered.head(200))

# EDA charts
st.header("Exploratory Data Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Age distribution")
    fig = px.histogram(filtered, x="Age", nbins=20, title="Age distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Workout frequency")
    fig = px.histogram(filtered, x="WorkoutFrequencyPerWeek", nbins=8, title="Workouts per week")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Preferred workout types")
    fig = px.pie(filtered, names="PreferredWorkoutType", title="Workout type share")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Relations: Steps vs Calories vs Sleep")
fig = px.scatter(filtered, x="AvgDailySteps", y="AvgDailyCalories", color="PreferredWorkoutType", size="WorkoutFrequencyPerWeek",
                 hover_data=["UserID","Age","Gender"], title="Steps vs Calories")
st.plotly_chart(fig, use_container_width=True)

# Prepare features for clustering
feats = ["Age","AvgDailySteps","AvgDailyCalories","WorkoutFrequencyPerWeek","SleepHours"]
st.write("Note: clustering uses numerical features:", feats)

# Preprocess: simple imputation and outlier capping performed here for dashboard use
df_proc = filtered.copy()

# Convert columns to numeric and coerce
for col in ["AvgDailySteps","AvgDailyCalories","SleepHours"]:
    df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

# Impute missing with median
for col in ["AvgDailySteps","AvgDailyCalories","SleepHours"]:
    med = df_proc[col].median()
    df_proc[col] = df_proc[col].fillna(med)

# Cap outliers using IQR (winsorize)
def cap_iqr(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return s.clip(lower=lo, upper=hi)

for col in ["AvgDailySteps","AvgDailyCalories","SleepHours"]:
    df_proc[col] = cap_iqr(df_proc[col])

# Run clustering if requested
if run_cluster:
    X = df_proc[feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Xs)
    df_proc['cluster'] = labels.astype(int)

    st.success(f"K-Means finished with K={n_clusters}")
    st.subheader("Cluster sizes")
    st.write(df_proc['cluster'].value_counts().sort_index())

    # PCA for 2D visualization
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(Xs)
    df_proc['pca1'] = pcs[:,0]
    df_proc['pca2'] = pcs[:,1]
    df_proc['pca3'] = pcs[:,2]

    # 2D scatter
    st.subheader("Clusters (PCA 2D)")
    fig2 = px.scatter(df_proc, x="pca1", y="pca2", color="cluster", hover_data=["UserID","Age","Gender","PreferredWorkoutType"],
                      title="Cluster projection (PCA)")
    st.plotly_chart(fig2, use_container_width=True)

    # 3D scatter
    st.subheader("Clusters (Interactive 3D)")
    fig3 = px.scatter_3d(df_proc, x="pca1", y="pca2", z="pca3", color="cluster", size="WorkoutFrequencyPerWeek",
                         hover_data=["UserID","Age","Gender","PreferredWorkoutType"], title="3D cluster projection")
    st.plotly_chart(fig3, use_container_width=True)

    # Cluster profiling
    st.subheader("Cluster profiling (means)")
    profile = df_proc.groupby("cluster")[feats + ["WorkoutFrequencyPerWeek"]].mean().round(2)
    st.dataframe(profile)

    # Recommendations per cluster (simple rule-based)
    st.subheader("Cluster recommendations (automated)")
    recs = []
    for cl in sorted(df_proc['cluster'].unique()):
        row = profile.loc[cl]
        rec = f"Cluster {cl}: "
        # heuristics
        if row['AvgDailySteps'] > df_proc['AvgDailySteps'].median() and row['AvgDailyCalories'] > df_proc['AvgDailyCalories'].median() and row['WorkoutFrequencyPerWeek'] >= 3:
            rec += "Active users — promote advanced challenges, premium features, in-app competitions."
        elif row['WorkoutFrequencyPerWeek'] <= 1 and row['AvgDailySteps'] < df_proc['AvgDailySteps'].median():
            rec += "Low-activity — send beginner programs, nudges, and 7-day starter plan to re-engage."
        elif row['SleepHours'] < 6.5:
            rec += "Poor sleepers — suggest recovery content, sleep-tracking features, and mindfulness content."
        else:
            rec += "Regular users — cross-sell classes / tailored content based on preferred workout type."
        recs.append(rec)
    for r in recs:
        st.markdown(f"- {r}")

    # Allow user to test a sample profile
    st.subheader("What cluster would a sample user belong to?")
    st.markdown("Enter sample features and click 'Predict cluster' to see the cluster and suggested action.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        age_sample = st.number_input("Age", 18, 90, 28)
    with c2:
        steps_sample = st.number_input("AvgDailySteps", 100, 30000, 7000)
    with c3:
        calories_sample = st.number_input("AvgDailyCalories", 800, 5000, 2200)
    with c4:
        wf_sample = st.number_input("Workout/wk", 0, 14, 3)
    with c5:
        sleep_sample = st.number_input("Sleep hours", 3.0, 12.0, 7.0, step=0.5)
    if st.button("Predict cluster for sample"):
        sample = np.array([[age_sample, steps_sample, calories_sample, wf_sample, sleep_sample]])
        sample_s = scaler.transform(sample)
        sample_label = int(kmeans.predict(sample_s)[0])
        st.success(f"Sample belongs to cluster {sample_label}")
        # show recommendation
        idx = int(sample_label)
        st.markdown(f"**Recommendation:** {recs[idx]}")

    # Download clustered data
    st.subheader("Download clustered dataset")
    outdf = df_proc.copy()
    outdf_fname = "fitness_users_clustered.csv"
    outdf.to_csv(outdf_fname, index=False)
    with open(outdf_fname, "rb") as f:
        st.download_button("Download CSV", data=f, file_name=outdf_fname, mime="text/csv")

else:
    st.info("Adjust filters and press 'Run K-Means' in the sidebar to calculate clusters.")
