import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("Mall_Customers.csv")

# REMOVE EXTRA SPACES FROM COLUMN NAMES
df.columns = df.columns.str.strip()

# =========================================================
# PROJECT TITLE
# =========================================================

print("\n===================================================")
print(" MARKETING ANALYTICS CUSTOMER SEGMENTATION PROJECT ")
print("===================================================\n")

# =========================================================
# FIRST 5 ROWS
# =========================================================

print("FIRST 5 ROWS:\n")
print(df.head())

# =========================================================
# DATASET SHAPE
# =========================================================

print("\n===================================================")
print("\nDATASET SHAPE:")
print(df.shape)

# =========================================================
# COLUMN NAMES
# =========================================================

print("\n===================================================")
print("\nCOLUMN NAMES:\n")
print(df.columns)

# =========================================================
# MISSING VALUES
# =========================================================

print("\n===================================================")
print("\nMISSING VALUES:\n")
print(df.isnull().sum())

# =========================================================
# DATA CLEANING
# =========================================================

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

# CLEAN INCOME COLUMN
df["Income"] = (
    df["Income"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
    .astype(float)
)

print("\n===================================================")
print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")

# =========================================================
# CONVERT EDUCATION TO NUMBERS
# =========================================================

education_mapping = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4
}

df["Education"] = df["Education"].map(education_mapping)

# =========================================================
# CREATE TOTAL SPENDING COLUMN
# =========================================================

df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)

# =========================================================
# GRAPH STYLE
# =========================================================

sns.set_style("darkgrid")

# =========================================================
# GRAPH 1 - CUSTOMER BIRTH YEAR DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Year_Birth"],
    bins=20,
    kde=True,
    color="blue"
)

plt.title("Customer Birth Year Distribution")
plt.xlabel("Year Birth")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("birth_year_distribution.png")

plt.show()

# =========================================================
# GRAPH 2 - INCOME DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Income"],
    bins=30,
    kde=True,
    color="green"
)

plt.title("Customer Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("income_distribution.png")

plt.show()

# =========================================================
# GRAPH 3 - EDUCATION DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="Education",
    data=df,
    hue="Education",
    palette="Set2",
    legend=False
)

plt.title("Education Distribution")
plt.xlabel("Education Level")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("education_distribution.png")

plt.show()

# =========================================================
# GRAPH 4 - MARITAL STATUS DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,5))

sns.countplot(
    x="Marital_Status",
    data=df,
    hue="Marital_Status",
    palette="Set3",
    legend=False
)

plt.title("Marital Status Distribution")
plt.xlabel("Marital Status")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("marital_status_distribution.png")

plt.show()

# =========================================================
# GRAPH 5 - INCOME VS SPENDING
# =========================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Income",
    y="Total_Spending",
    data=df,
    color="purple"
)

plt.title("Income vs Total Spending")
plt.xlabel("Income")
plt.ylabel("Total Spending")

plt.tight_layout()

plt.savefig("income_vs_spending.png")

plt.show()

# =========================================================
# STANDARDIZATION
# =========================================================

X = df[["Income", "Total_Spending"]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# ELBOW METHOD
# =========================================================

wcss = []

for i in range(1,11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o",
    color="red"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.tight_layout()

plt.savefig("elbow_method.png")

plt.show()

# =========================================================
# KMEANS CLUSTERING
# =========================================================

kmeans = KMeans(
    n_clusters=4,
    init="k-means++",
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# =========================================================
# GRAPH 6 - CUSTOMER SEGMENTS
# =========================================================

plt.figure(figsize=(10,7))

sns.scatterplot(
    x=df["Income"],
    y=df["Total_Spending"],
    hue=df["Cluster"],
    palette="Set1",
    s=100
)

plt.title("Customer Segmentation using KMeans Clustering")
plt.xlabel("Income")
plt.ylabel("Total Spending")

plt.tight_layout()

plt.savefig("customer_segments.png")

plt.show()

# =========================================================
# GRAPH 7 - CLUSTER DISTRIBUTION
# =========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    x="Cluster",
    data=df,
    hue="Cluster",
    palette="Set2",
    legend=False
)

plt.title("Customer Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("cluster_distribution.png")

plt.show()

# =========================================================
# GRAPH 8 - OUTLIER DETECTION
# =========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Income"],
    color="orange"
)

plt.title("Outlier Detection in Income")

plt.tight_layout()

plt.savefig("income_outliers.png")

plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n===================================================")
print(" PROJECT SUMMARY ")
print("===================================================\n")

print("1. Dataset Loaded Successfully")
print("2. Data Cleaning Completed")
print("3. Exploratory Data Analysis Completed")
print("4. Income & Spending Analysis Completed")
print("5. Outlier Detection Completed")
print("6. Elbow Method Applied")
print("7. KMeans Clustering Applied")
print("8. Customer Segments Identified")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
