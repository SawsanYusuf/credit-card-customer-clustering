# Credit Card Customer Clustering Analysis

## Project Overview

This project aims to analyze and segment credit card users based on their financial behavior and transaction patterns. By leveraging **Machine Learning (K-Means Clustering)**, we identified distinct customer personas to optimize product offerings and marketing strategies for financial institutions.


## Dataset Overview

The dataset is sourced from **Kaggle** ([Dataset](https://www.kaggle.com/datasets/hassanraof/cc-general-csv)) and contains the usage behavior of approximately **8,950 active credit card holders**.

* **Data Structure:** 18 features (17 numerical variables + 1 unique identifier `CUST_ID`).
* **Key Features:** `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, and `PAYMENTS`.


## Project Steps

### 1. Data Exploration & Cleaning

* **Missing Values:** Found minimal missing data in `MINIMUM_PAYMENTS` (3.49%) and `CREDIT_LIMIT` (0.01%).
* **Imputation:** Missing values were filled using **Mean Imputation** to maintain the dataset size and statistical integrity.
* **Outlier Detection:** Used Boxplots to identify significant skewness in spending and credit limits.

![Distribution of Credit Cards Limit](https://github.com/SawsanYusuf/credit-card-customer-clustering/blob/main/Images/newplot.png)

### 2. Feature Selection (Variance Analysis)

To improve clustering performance, we focused on features with high discriminative power:

* **Trimmed Variance:** Calculated the trimmed variance to identify features that represent typical behavior while reducing the noise from extreme outliers.
* **Final Features:** Selected the **Top 5** features: `PURCHASES`, `CASH_ADVANCE`, `PAYMENTS`, `BALANCE`, and `CREDIT_LIMIT`.

| High Variance (Raw) | Trimmed Variance (Selected) |
| --- | --- |
|  | ![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/high%20variance.png)

### 3. Model Development & Evaluation

* **Scaling:** Applied `StandardScaler` to normalize the data before clustering.
* **K-Means Tuning:** Iterated through K values (2 to 12) to find the optimal balance between **Inertia** and **Silhouette Score**.
* **Final Decision:** Selected **$K=3$** as the optimal number of clusters for business interpretability.
* **Visualization:** Used **PCA (Principal Component Analysis)** to represent the 3D clusters in a 2D space.

##  Visualizing & Validating Clusters
Since the visualizations were created using **Plotly Express**, which may not render directly in the GitHub Notebook view, they are provided below with a detailed analysis:

### **1. Why $K=3$ Clusters? (Numerical Validation)**
The decision to finalize the model with 3 clusters was based on a comparative analysis of **Inertia** and **Silhouette Scores**:
* **Inertia Analysis:** We observed the steepest drop (the "Elbow") when moving from $K=2$ to $K=3$ (from **31,229** to **25,753**).
* **Silhouette Score Analysis:** Although $K=2$ had a higher score (**0.549**), the score for $K=3$ remained robust at **0.495**. 
* **The Trade-off:** Choosing $K=3$ provided the most balanced "Business Logic," allowing a clear distinction between the "Elite", "Affluent", and "Everyday" users.

### **2. Cluster Representation & Distribution**
The following plots demonstrate the clear separation and financial behavior of the identified segments:

| **PCA Cluster Separation** | **Mean Financials per Cluster** |
| :---: | :---: |
| ![PCA Representation of Clusters](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/PCA.png)
 | ![Mean Household Finances by Cluster](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/clusters.png) 


* **Cluster 1:** High spending and high payment activity (The Elite).
* **Cluster 0:** Balanced spending with stable credit limits (The Affluent).
* **Cluster 2:** Routine, low-value daily transactions (Everyday Users).


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
1.  **Hyper-Personalized Marketing**: Tailoring campaigns to the specific behavioral DNA of each cluster.
2.  **Risk Mitigation**: Aligning credit limits with verified income brackets and spending patterns.
3.  **Product Optimization**: Designing tier-specific features aligned with real behavioral clusters rather than assumptions.

هيك تص


## Technologies Used

* **Language:** Python 🐍
* **Libraries:** Pandas, NumPy, Scikit-learn (KMeans, PCA), Plotly, Matplotlib, Seaborn.
* **Environment:** Jupyter Notebook / Kaggle.

## Author
**Sawsan Yousef** 

*Data Scientist | Predictive Modeling | Computer Vision*

[LinkedIn](https://www.linkedin.com/in/sawsan-yusuf-027b2b214?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) | [Medium](https://medium.com/@sawsanyusuf)













      
      
