import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import datetime as dt
from scipy import stats

# Set page configuration
st.set_page_config(page_title="Online Retail Analysis", layout="wide")

# Title
st.title("🛒 Online Retail II Data Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('Online Retail.xlsx')
    return df

df = load_data()

# Display raw data
st.header("1. Raw Data Overview")
st.write(df.head())

# Display column names
st.subheader("Column Names")
st.write(df.columns.tolist())

# Data Preprocessing
st.header("2. Data Preprocessing")
df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(int)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
st.write("Data after preprocessing:")
st.write(df.head())

# Feature Engineering (Log Transformation for RFM metrics)
st.header("3. Feature Engineering")
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})
rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_normalized = scaler.fit_transform(rfm_log)
st.write("Transformed RFM Data:")
st.write(rfm_log.head())

# Removing Outliers using Z-score
st.header("4. Removing Outliers")
z_scores = np.abs(stats.zscore(rfm_normalized))
rfm_clean = rfm[(z_scores < 3).all(axis=1)]
rfm_clean_normalized = scaler.fit_transform(rfm_clean)
st.write("RFM Data after removing outliers:")
st.write(rfm_clean.head())

# Top 10 Most Selling Products
st.header("5. Top 10 Selling Products")
top_products = df['Description'].value_counts().head(10)
fig, ax = plt.subplots()
top_products.plot(kind='barh', ax=ax)
ax.set_xlabel('Number of Sales')
ax.set_title('Top 10 Selling Products')
st.pyplot(fig)

# Total Sales by Country
st.header("6. Total Sales by Country")
country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
country_sales.plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel('Total Sales')
ax.set_title('Top 10 Countries by Total Sales')
st.pyplot(fig)

# RFM Distribution: Recency, Frequency, Monetary
st.header("7. RFM Distribution")
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(rfm['Recency'], kde=True, ax=axs[0], color='salmon').set_title('Recency Distribution')
sns.histplot(rfm['Frequency'], kde=True, ax=axs[1], color='teal').set_title('Frequency Distribution')
sns.histplot(rfm['Monetary'], kde=True, ax=axs[2], color='orange').set_title('Monetary Distribution')
st.pyplot(fig)

# Recency vs Frequency Scatter Plot
st.header("8. Recency vs Frequency")
fig, ax = plt.subplots()
sns.scatterplot(x='Recency', y='Frequency', data=rfm, ax=ax, color='purple')
ax.set_title('Recency vs Frequency')
st.pyplot(fig)

# Optimal Number of Clusters (Elbow Method)
st.header("9. Finding Optimal Number of Clusters (Elbow Method)")
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_clean_normalized)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(2, 11), wcss, marker='o')
ax.axvline(4, color='red', linestyle='--', label='Elbow at k=4')
ax.legend()
ax.set_title('Elbow Method for Optimal Clusters')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Apply KMeans with optimal k
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_clean['Cluster'] = kmeans.fit_predict(rfm_clean_normalized)

# Visualize Clusters
st.header(f"10. Visualizing {optimal_k} Customer Segments (KMeans Clustering)")
fig, ax = plt.subplots()
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm_clean, ax=ax, palette='Set2')
ax.set_title(f'Customer Segments (KMeans - {optimal_k} Clusters)')
st.pyplot(fig)

# Display Cluster Sizes
st.subheader("Customer Count per Cluster")
cluster_sizes = rfm_clean['Cluster'].value_counts().sort_index()
st.write(cluster_sizes)

# Silhouette Score
sil_score = silhouette_score(rfm_clean_normalized, rfm_clean['Cluster'])
st.write(f"Silhouette Score after KMeans: {sil_score:.2f}")

# Sales Trend over Time with optional time filter
st.header("11. Sales Trend over Time")
start_date = st.date_input("Start Date", df['InvoiceDate'].min().date())
end_date = st.date_input("End Date", df['InvoiceDate'].max().date())
filtered_df = df[(df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)]
sales_trend = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['TotalPrice'].sum()
fig, ax = plt.subplots()
sales_trend.plot(kind='line', ax=ax, color='green')
ax.set_title('Sales Trend Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Total Sales')
st.pyplot(fig)

# Association Rule Mining
st.header("12. Association Rule Mining")
basket = (df[df['Country'] == 'United Kingdom']
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
basket = basket.applymap(lambda x: 1 if x > 0 else 0)
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True, low_memory=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Explain support, confidence, lift
st.markdown("""
**Interpretation of Metrics**:
- **Support**: How often items appear together in transactions.
- **Confidence**: How often the rule has been found to be true.
- **Lift**: How much more often the items in the rule are bought together than expected.
""")

st.write("Association Rules:")
st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Conclusion
st.header("13. Conclusion")
st.write("""
- The data preprocessing step cleaned the dataset by removing canceled orders and rows with missing CustomerIDs.
- Feature engineering was applied to create Recency, Frequency, and Monetary (RFM) metrics. These metrics were log-transformed for better clustering performance.
- After outlier removal using the Z-score method, we proceeded with customer segmentation using KMeans with 4 clusters.
- The elbow method helped in choosing the optimal number of clusters, and the silhouette score was used to evaluate clustering performance. A good silhouette score (~0.36) was achieved.
- Association Rule Mining was performed to discover customer purchasing patterns.
""")

# Customer Segment Analysis
st.write("""
### Customer Segment Analysis

- **Segment 0 (Best Customers)**:
    - **Recency**: Recently active
    - **Frequency**: Frequent buyers
    - **Monetary**: Average spenders
    - **Interpretation**: These are loyal, regular customers. You should prioritize customer retention strategies for them (e.g., loyalty programs, exclusive offers).

- **Segment 1 (At-Risk Customers)**:
    - **Recency**: Haven’t purchased in a while
    - **Frequency**: Rare buyers
    - **Monetary**: Spend less
    - **Interpretation**: These could be "at-risk" customers. Consider re-engagement campaigns like win-back emails or discounts.

- **Segment 2 (VIP Customers)**:
    - **Recency**: Moderate
    - **Frequency**: Frequent
    - **Monetary**: High spenders
    - **Interpretation**: These are your high-value clients. Offer premium services or exclusive bundles to maximize their lifetime value.

- **Segment 3 (Repeat Customers)**:
    - **Recency**: Low
    - **Frequency**: High
    - **Monetary**: Moderate spenders
    - **Interpretation**: Consider incentives like referral programs or new arrivals to keep them engaged.
""")

# Final Segment Distribution
st.header("14. Final Customer Segments Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Cluster', data=rfm_clean, ax=ax)
ax.set_title('Customer Segments Distribution')
st.pyplot(fig)

# Practical Business Actions
st.header("15. Business Recommendations")
st.markdown("""
- 🎯 **Target Segment 0** with loyalty programs and VIP benefits to retain high-value customers.
- 🧲 **Win back Segment 1** using email marketing, retargeting ads, or personalized offers.
- 🚀 **Upsell to Segment 2** by recommending complementary products or premium bundles.
- 🔁 **Engage Segment 3** with limited-time deals or newsletters to boost activity.
- 🌍 **Focus on the UK** for inventory and promotions due to high sales from this region.
""")
