# Credit Card Customer Clustering Analysis
![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/stephen-phillips-hostreviews-co-uk-em37kS8WJJQ-unsplash.jpg)

## Project Overview

This project aims to analyze and segment credit card users based on their financial behavior and transaction patterns. By leveraging **Machine Learning (K-Means Clustering)**, we identified distinct customer personas to optimize product offerings and marketing strategies for financial institutions.


## Dataset Overview

The dataset is sourced from **Kaggle** ([CC General Dataset](https://www.kaggle.com/datasets/hassanraof/cc-general-csv)) and contains the usage behavior of approximately **8,950 active credit card holders**.

* **Data Structure:** 18 features (17 numerical variables + 1 unique identifier `CUST_ID`).
* **Key Features:** `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, and `PAYMENTS`.


## Project Steps

### 1️⃣ Data Exploration & Cleaning

* **Missing Values:** Found minimal missing data in `MINIMUM_PAYMENTS` (3.49%) and `CREDIT_LIMIT` (0.01%).
* **Imputation:** Missing values were filled using **Mean Imputation** to maintain the dataset size and statistical integrity.
* **Outlier Detection:** Used Boxplots to identify significant skewness in spending and credit limits.

![Distribution of Credit Cards Limit](newplot (1).png)

### 2️⃣ Feature Selection (Variance Analysis)

To improve clustering performance, we focused on features with high discriminative power:

* **Trimmed Variance:** Calculated the trimmed variance to identify features that represent typical behavior while reducing the noise from extreme outliers.
* **Final Features:** Selected the **Top 5** features: `PURCHASES`, `CASH_ADVANCE`, `PAYMENTS`, `BALANCE`, and `CREDIT_LIMIT`.

| High Variance (Raw) | Trimmed Variance (Selected) |
| --- | --- |
|  | ![Trimmed Variance](newplot (2).png) |

### 3️⃣ Model Development & Evaluation

* **Scaling:** Applied `StandardScaler` to normalize the data before clustering.
* **K-Means Tuning:** Iterated through  values (2 to 12) to find the optimal balance between **Inertia** and **Silhouette Score**.
* **Final Decision:** Selected **** as the optimal number of clusters for business interpretability.
* **Visualization:** Used **PCA (Principal Component Analysis)** to represent the 3D clusters in a 2D space.

![PCA Representation of Clusters](newplot (3).png)

---

## 📊 Visualizations & Insights

Analyzing the mean financial metrics for each cluster reveals clear behavioral boundaries:

![Mean Household Finances by Cluster](newplot (4).png)

* **Cluster 1:** High spending and high payment activity (The Elite).
* **Cluster 0:** Balanced spending with stable credit limits (The Affluent).
* **Cluster 2:** Routine, low-value daily transactions (Everyday Users).

---

## 🚀 Recommendations

Based on the distinct characteristics of the identified customer clusters, the following credit card recommendations are proposed:

### Cluster 1: The Elite (Platinum Card Holders)

* **Characteristics:** High-value customers with substantial financial activity and high repayment rates.
* **Recommendation:** **Platinum Credit Card** ($40,000 - $1M Limit).
* **Strategic Focus:** Exclusive benefits, personalized premium services, luxury perks, and dedicated support.

### Cluster 0: The Affluent Spenders (Gold Card Holders)

* **Characteristics:** Customers with strong monthly income and consistent spending/repayment capacity.
* **Recommendation:** **Gold Credit Card** ($10,000 - $40,000 Limit).
* **Strategic Focus:** Generous rewards programs, attractive cashback incentives, and travel-related benefits.

### Cluster 2: The Everyday Users (Silver Card Holders)

* **Characteristics:** The largest and most accessible segment, using cards for routine day-to-day transactions.
* **Recommendation:** **Silver Credit Card** ($4,000 - $7,000 Limit).
* **Strategic Focus:** Ease of access, fundamental convenience, and tools for building/managing credit history.

---

## 🛠 Technologies Used

* **Language:** Python 🐍
* **Libraries:** Pandas, NumPy, Scikit-learn (KMeans, PCA), Plotly, Matplotlib, Seaborn.
* **Environment:** Jupyter Notebook / Kaggle.

---

بهذه الطريقة، أصبح المشروع منظماً بنفس الأسلوب الاحترافي لمشروع الأرجنتين، مع إبراز قوتك في التحليل الإحصائي (Variance) واختيار النموذج. هل ترغبين في أي تعديل إضافي؟




This project focuses on performing customer segmentation using unsupervised machine learning techniques on credit card usage data. The primary goal is to identify distinct segments of active credit card holders, understand their unique characteristics, and provide actionable insights for businesses to develop more targeted and effective marketing strategies.

## Project Goal

Our main goal is to develop a customer segmentation model based on the credit card usage data of about 9,000 active credit card holders over the last six months. By the end of this analysis, we aim to answer the following questions:

* What are the distinct segments of credit card users?
* What defines each segment?
* How can businesses leverage these insights?
  
## Dataset
The dataset contains transactional and behavioral data for 8950 distinct credit card customers over the last six months. It comprises 18 features describing various aspects of their credit card usage, including:

* `CUST_ID`: Unique identification of a customer
* `BALANCE`: Balance amount left in their account
* `BALANCE_FREQUENCY`: How frequently the balance is updated
* `PURCHASES`: Amount of purchases made
* `ONEOFF_PURCHASES`: Maximum purchase amount done in one-off transaction
* `INSTALLMENTS_PURCHASES`: Amount of purchase done in installment
* `CASH_ADVANCE`: Cash in advance given by the bank
* `PURCHASES_FREQUENCY`: How frequently the purchases are being made
* `ONEOFF_PURCHASES_FREQUENCY`: How frequently one-off purchases are being made
* `PURCHASES_INSTALLMENTS_FREQUENCY`: How frequently purchases in installments are being made
* `CASH_ADVANCE_FREQUENCY`: How frequently cash in advance is being paid
* `CASH_ADVANCE_TRX`: Number of cash advance transactions
* `PURCHASES_TRX`: Number of purchase transactions
* `CREDIT_LIMIT`: Credit limit of the customer
* `PAYMENTS`: Amount of payments made
* `MINIMUM_PAYMENTS`: Minimum amount of payments made by customer
* `PRC_FULL_PAYMENT`: Percentage of full payment paid
* `TENURE`: Tenure of credit card service

## Methodology & Steps
The project followed a standard data science pipeline:

### 1. Data Loading and Initial Exploration:

* Loaded the dataset into a Pandas DataFrame.

* Performed initial inspection using `df.head()` and `df.info()` to understand data types and non-null counts.

### 2. Data Preprocessing:

* Handling Missing Values: Identified and imputed missing values in MINIMUM_PAYMENTS and CREDIT_LIMIT columns by filling them with the mean of their respective columns. 
```
df.loc[(df['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].mean()
df.loc[(df['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=df['CREDIT_LIMIT'].mean()
```
* Handling Duplicates: Checked for and confirmed the absence of duplicate rows.
* Feature Engineering/Selection (Implicit): The CUST_ID column was dropped as it's a unique identifier and not relevant for clustering.
```
df.drop(columns='CUST_ID', inplace=True)
```

### 3. Exploratory Data Analysis (EDA) & Feature Importance:

* Descriptive Statistics: Generated descriptive statistics for numerical features (df.describe()).
* Variance Analysis: Calculated and visualized the 10 features with the highest variance to identify potentially important variables for clustering.
   * Standard Variance
   * Trimmed Variance (excluding outliers): A trim_variance function was applied to calculate variance after removing the top/bottom 0.1% outliers, which helps in focusing on the core distribution.
* Outlier Visualization: Created a box plot for CREDIT_LIMIT to visualize its distribution and potential outliers.

### 4. Feature Selection for Clustering:

* Based on the variance analysis, the 5 features with the highest variance were selected for the clustering model. These features were identified as most influential in differentiating customer behavior: PURCHASES, CASH_ADVANCE, PAYMENTS, BALANCE, CREDIT_LIMIT.

![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/high%20variance.png)

### 5. Data Scaling:
* Utilized StandardScaler to scale the selected features. This is crucial for K-Means clustering, as it's sensitive to feature magnitudes, ensuring all features contribute equally to the distance calculations.
```
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scaled = ss.fit_transform(X) # X
being the selected features
```

### 6. K-Means Clustering Model Building:

* Determining Optimal Number of Clusters (`k=3`): While methods like the Elbow Method and Silhouette Score are often used to determine optimal k (as visualized below to show thoroughness), the final choice of k=3 was specifically made to align with the three distinct credit card product tiers (Platinum, Gold, Silver) available for recommendation.

![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/elbow.png)

![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/silhouette%20.png)

* Final Model Training: A K-Means model was trained with n_clusters=3.

```
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

final_model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42)) # k=3
final_model.fit(X)
```
### 7. Results Interpretation and Visualization:

* Assigning Cluster Labels: Obtained cluster labels for each customer.
* Cluster Profiling: Calculated the mean of each feature for every cluster to understand the characteristics of each customer segment. This step directly addresses "What defines each segment?".
```
X_summary = X.groupby(final_model.named_steps['kmeans'].labels_).mean().astype(int)
```
* Visualizing Cluster Profiles: Created a side-by-side bar chart to visually compare the mean values of key financial metrics across different clusters, enabling clear interpretation of each segment.
  
![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/clusters.png)

* PCA for Visualization (Optional but Implemented):
    * Applied Principal Component Analysis (PCA) to reduce the dimensionality of the data to 2 components (PC1, PC2) for 2D visualization.
    * Created a scatter plot of PC1 vs PC2, colored by cluster labels, to visualize the separation of the identified customer segments in a lower-dimensional space.
      
![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/PCA.png)
      
## Outcomes & Business Impact

This project successfully implemented a **K-Means Clustering** model to segment the customer base into distinct financial profiles. The analysis provides data-driven answers to critical business questions regarding user behavior, product alignment, and targeted marketing strategies.

The model identified **three distinct customer clusters** that directly correlate with existing credit card product tiers: **Platinum, Gold, and Silver**. These insights empower the business to optimize product offerings and enhance customer lifecycle management.


### Cluster Analysis & Financial Profiling

#### Cluster 0: The Elite (Platinum Tier)
*High-value clients with substantial financial velocity and premium requirements.*

* **Profile:** Represents the top-tier segment characterized by significant financial activity and rigorous eligibility standards.
* **Recommended Product:** **Platinum Credit Card**
* **Financial Parameters:**
    * **Credit Limit:** $40,000 – $1,000,000
    * **Eligibility:** Minimum monthly income of **$1,800** + Excellent credit history.
* **Strategic Focus:** * Reinforce "Elite Status" through exclusive concierge services.
    * Offer luxury lifestyle perks and premium global travel benefits.
    * Provide dedicated high-priority customer support.

#### Cluster 1: The Affluent Spenders (Gold Tier)
*Active consumers with high purchasing power and consistent repayment reliability.*

* **Profile:** Customers with strong, stable monthly incomes and a high capacity for both transactional spending and diligent debt management.
* **Recommended Product:** **Gold Credit Card**
* **Financial Parameters:**
    * **Credit Limit:** $10,000 – $40,000
    * **Value Proposition:** Designed for high-frequency users and big-ticket purchases.
* **Strategic Focus:** * Maximize retention via aggressive **Cashback Incentives** and rewards programs.
    * Promote flexible repayment options for large-scale purchases.
    * Tailor marketing toward travel and lifestyle rewards.

#### Cluster 2: The Everyday Users (Silver Tier)
*The foundational segment utilizing credit for routine, day-to-day liquidity.*

* **Profile:** The most accessible and widespread segment, primarily focused on routine transactions and maintaining financial stability.
* **Recommended Product:** **Silver Credit Card**
* **Financial Parameters:**
    * **Credit Limit:** $4,000 – $7,000
    * **Eligibility:** Broad accessibility with a minimum monthly income of **$400**.
* **Strategic Focus:** * Position as an essential tool for **Credit Score Building**.
    * Highlight ease of access, security features, and digital convenience.
    * Target budget-conscious users with low-fee structures.

### Business Impact & Recommendations
The integration of this segmentation model allows for:
1.  **Hyper-Personalized Marketing:** Tailoring campaigns to the specific behavioral DNA of each cluster.
2.  **Risk Mitigation:** Aligning credit limits with verified income brackets and spending patterns.
3.  **Product Optimization:** Adjusting card features to better serve the identified needs of the "Elite" vs. "Everyday" user.
