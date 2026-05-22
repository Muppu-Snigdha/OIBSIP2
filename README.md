# 🧠 Marketing Analytics Customer Segmentation Project

## 📌 Project Overview

This project focuses on **Customer Segmentation using Machine Learning** and **Marketing Analytics** techniques.

The main objective of this project is to group customers into different segments based on:

- Income
- Spending Behaviour
- Purchasing Patterns

The project uses:

- Exploratory Data Analysis (EDA)
- Data Visualization
- Data Cleaning
- KMeans Clustering Algorithm

This project was developed as part of the **Oasis Infobyte Data Science Internship**.


# 📂 Dataset Information

Dataset Used:

- Mall_Customers.csv / marketing_data.csv

Dataset Features:

- ID
- Year_Birth
- Education
- Marital_Status
- Income
- Kidhome
- Teenhome
- Recency
- Spending Data
- Purchase Information
- Country



# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn



# 📊 Project Workflow

## 1️⃣ Data Loading

Dataset loaded using Pandas.
python
df = pd.read_csv("marketing_data.csv")


2️⃣ Data Cleaning

Performed:

Removed duplicate values

Removed missing values

Cleaned income column

Converted categorical data into numerical values


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)



3️⃣ Exploratory Data Analysis (EDA)

Graphs were generated for:

Customer Birth Year Distribution

Income Distribution

Education Distribution

Marital Status Distribution

Income vs Spending

Outlier Detection





4️⃣ Feature Engineering

Total spending calculated using:

df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)




5️⃣ Data Standardization

StandardScaler was applied before clustering.

scaler = StandardScaler()

6️⃣ Elbow Method

The Elbow Method was used to determine the optimal number of clusters.

KMeans()



7️⃣ KMeans Clustering

Customers were divided into different groups based on:

Income

Spending Behaviour


kmeans = KMeans(n_clusters=4)



📈 Graphs Generated

The project automatically generates and saves graphs.

Graph Name	File Name

Birth Year Distribution	birth_year_distribution.png
Income Distribution	income_distribution.png
Education Distribution	education_distribution.png
Marital Status Distribution	marital_status_distribution.png
Income vs Spending	income_vs_spending.png
Elbow Method	elbow_method.png
Customer Segmentation	customer_segmentation.png
Cluster Distribution	cluster_distribution.png
Income Outlier Detection	income_outlier_detection.png



🤖 Machine Learning Algorithm Used

KMeans Clustering

KMeans is an unsupervised Machine Learning algorithm used to group customers into clusters based on similarities.

The algorithm helps businesses:

Identify target customers

Improve marketing strategies

Understand customer behaviour

Increase sales performance




▶️ How To Run The Project

Step 1

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Step 2

Place these files inside the same folder:

marketing_data.csv

customer_segmentation.py


Step 3

Run the project:

python customer_segmentation.py



✅ Final Output

After successful execution:

Graphs display automatically

PNG graph files save automatically

Customer clusters generated successfully

Final project summary displayed




📌 Project Insights

Using Customer Segmentation:

High-value customers can be identified

Businesses can improve marketing campaigns

Spending patterns become easier to analyze

Companies can make better business decision

📌 Conclusion

This project demonstrates how Machine Learning and Data Analytics can be used to analyze customer behaviour and create meaningful customer segments.

The project improves practical skills in:

Data Cleaning

Data Visualization

Exploratory Data Analysis

Machine Learning

Customer Analytics

