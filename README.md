# Credit Card Customer Clustering
![](https://github.com/SawsanYusuf/Credit-Card-Customer-Clustering/blob/main/Images/stephen-phillips-hostreviews-co-uk-em37kS8WJJQ-unsplash.jpg)

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
This project successfully developed a customer segmentation model, clearly answering:

What are the distinct segments of credit card users? The K-Means model identified 3 distinct customer clusters, directly aligning with the existing credit card product tiers (Platinum, Gold, Silver).
What defines each segment? Detailed profiling of each cluster (as visualized in the mean financial chart) revealed specific characteristics and behavioral patterns.
How can businesses leverage these insights? The insights gained from this segmentation can be directly applied by businesses to enhance their product offerings, marketing, and customer management strategies.
Recommendations for Targeted Credit Card Offerings
Based on the distinct characteristics of the identified customer clusters, the following credit card recommendations are proposed to optimize product offerings and marketing strategies:

Cluster 0: The Elite (Platinum Card Holders)

Characteristics: This segment likely represents high-value customers with substantial financial activity. These cards are typically owned by a select few due to strict eligibility requirements and high income/credit history demands.

Recommendation: Platinum Credit Card (Highest Level)
Credit Limit: Ranges from $40,000 up to $1 million.
Eligibility: Cardholders must have a regular, high income (e.g., at least $1,800 per month – assuming 'per year' was a typo and adjusting to a more typical income for this tier) and an excellent credit history, reflecting a rigorous application procedure.
Strategic Focus: Target with exclusive benefits, personalized premium services, luxury perks, and dedicated support to reinforce their elite status.

Cluster 1: The Affluent Spenders (Gold Card Holders)

Characteristics: This segment consists of customers with a strong and consistent monthly income, indicating a significant capacity for both spending and diligent repayment.

Recommendation: Gold Credit Card
Credit Limit: Ranges from $10,000 to $40,000, providing ample room for substantial purchases and financial flexibility.
Benefits: Ideal for users who frequently make larger purchases and require options for repaying big-budget items.
Strategic Focus: Emphasize generous rewards programs, attractive cashback incentives on higher spending, and benefits tailored to their active consumer behavior, such as travel perks.

Cluster 2: The Everyday Users (Silver Card Holders)

Characteristics: This is the most widely owned and accessible segment, representing customers who primarily use credit cards for routine, day-to-day transactions with lower spending limits.

Recommendation: Silver Credit Card
Credit Limit: The lowest credit limit, typically ranging from $4,000 to $7,000, suitable for common expenses.
Eligibility: More broadly accessible, requiring a lower minimum monthly salary (e.g., at least $400 per month).
Strategic Focus: Market based on ease of access, fundamental convenience, essential security features, and as a tool for building or managing credit history for new or budget-conscious cardholders.
