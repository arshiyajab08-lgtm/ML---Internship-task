import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍 Customer Segmentation Dashboard")

data = pd.read_csv("Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

st.sidebar.header("Model Settings")
k = st.sidebar.slider("Number of Clusters", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

data["Cluster"] = labels
sil = silhouette_score(X, labels)

# ---- Metrics ----
c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(data))
c2.metric("Clusters", k)
c3.metric("Silhouette Score", round(sil, 3))

# ---- Plot ----
fig, ax = plt.subplots()
scatter = ax.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=labels
)

ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker="X"
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Clusters")

st.pyplot(fig)

st.subheader("📊 Cluster-wise Summary")
st.write(data.groupby("Cluster").mean(numeric_only=True))
