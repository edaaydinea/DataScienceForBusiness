# DataScienceForBusiness  

This repository includes six real-world case studies in data science applied to business scenarios.  

## Case Study 1: Employee Attrition Prediction  

### **Overview**  

Employee attrition is a significant challenge for organizations, leading to high costs and reduced productivity. This case study focuses on using machine learning to predict employee attrition and empower HR teams to make proactive, data-driven decisions.  

### **Problem Statement**  

- **High Costs of Hiring**: Recruiting a new employee costs 15%-20% of their salary, with an average of $7,645 for small companies.  
- **Retention Challenges**: Understanding factors like job satisfaction and work-life balance is critical.  
- **Need for Prediction**: An ML-based model is required to identify employees likely to quit.  

### **Business Impact**  

1. **Cost Savings**: Minimize hiring costs and revenue losses.  
2. **Increased Productivity**: Retain top talent and reduce turnover.  
3. **Strategic Planning**: Optimize workforce management using model insights.  

### **Dataset Features**  

- Includes variables such as job involvement, education, job satisfaction, performance ratings, and work-life balance.  

### **Approach**  

- Models like logistic regression, random forests, and neural networks will be used to predict attrition and support HR strategies.  

Explore this [case study](https://github.com/edaaydinea/DataScienceForBusiness/blob/main/1.%20Human%20Resources%20Data/Human_Resources_Department.ipynb) to see how data science can transform HR challenges into actionable solutions!  

## Case Study 2: Customer Segmentation for Enhanced Marketing Strategies

### **Overview**

This case study focuses on segmenting credit card customers to provide actionable insights for targeted marketing strategies. By analyzing customer credit card usage and behavior, distinct customer groups are identified using the K-Means clustering algorithm. This segmentation enables the business to tailor marketing campaigns, personalize offers, and improve overall customer satisfaction.

### **Problem Statement**

- **Relevant Problem 1**: In a competitive credit card market, a one-size-fits-all marketing approach leads to inefficiencies. The business impact includes wasted marketing spend, low customer engagement, and missed opportunities for targeted promotions. The main issue is the inability to personalize customer interactions effectively.
- **Relevant Problem 2**: The company faces challenges in understanding the diverse behaviors and preferences of its customer base. This lack of insight hinders the ability to offer relevant products and services, reducing customer satisfaction and loyalty.
- **Solution Needed**: An ML-based model can address these problems by clustering customers into homogeneous groups based on their transactional and behavioral attributes. This segmentation will enable targeted marketing, personalized recommendations, and proactive customer retention strategies.

### **Business Impact**

1. **Business Benefit 1**: Targeted marketing campaigns will improve customer engagement and increase conversion rates.
2. **Business Benefit 2**: Personalized offers and recommendations will enhance customer satisfaction and loyalty.
3. **Business Benefit 3**: Optimized resource allocation will maximize marketing ROI and profitability.

### **Dataset Features**

The dataset includes key features such as:

- `BALANCE`: Current balance on the credit card, reflecting the customer's spending and credit utilization.
- `PURCHASES`: Total purchases made by the customer, indicating spending behavior.
- `CASH_ADVANCE`: Total cash advances taken, highlighting liquidity needs.
- `CREDIT_LIMIT`: Credit limit assigned, showing creditworthiness.
- `PAYMENTS`: Total payments made, indicating payment behavior.
- These features are crucial for identifying distinct customer segments with similar spending patterns, credit usage, and financial behavior, enabling the model to make accurate and relevant predictions.

### **Approach**

The K-Means clustering algorithm was selected to segment customers based on their credit card usage patterns. K-Means is effective in partitioning datasets into distinct clusters by minimizing the variance within each cluster. This method was chosen for its simplicity, efficiency, and ability to handle large datasets, making it suitable for identifying meaningful customer segments.

[Click here to explore the case study](https://github.com/edaaydinea/DataScienceForBusiness/blob/main/2.%20Marketing%20Department/Marketing_Department.ipynb)

## Case Study 3: Sales Forecasting for Retail Stores

### **Overview**

This case study addresses the challenge retail companies face in accurately forecasting fluctuating customer demand and sales trends across numerous stores. The primary problem is to develop a robust model capable of forecasting future daily sales for individual stores with high accuracy, using historical sales data (`train.csv`) and store-specific features (`store.csv`). The machine learning methodology involves detailed data analysis, preprocessing, and the evaluation and selection of various time-series and machine learning models (e.g., ARIMA, Prophet, XGBoost, LightGBM, Deep Learning models) to predict sales.

### **Problem Statement**

-   **Demand Fluctuations**: Sales are influenced by numerous factors, including promotions, holidays (both state and school holidays), seasonal effects, weekends, and whether stores are open or closed. Modeling this variability is complex. The business impact includes inventory mismanagement (overstocking or understocking) and suboptimal staff scheduling.
-   **Store Heterogeneity**: Each store has unique characteristics (e.g., `StoreType`, `Assortment`, `CompetitionDistance`). Understanding and incorporating the impact of these features on sales is crucial. This leads to challenges in ineffective marketing and promotions if not addressed.
-   **Data Complexity & Scalability**: The nature of time-series data, potential missing information, and outliers can complicate model development. Furthermore, the solution must be scalable to generate forecasts for a large number of stores. This impacts cash flow planning if not managed effectively.
-   **Solution Needed**: An ML-based model can address these problems by forecasting daily sales for each store, utilizing historical sales data, store characteristics, promotional information, and holiday data to significantly improve forecast accuracy and provide actionable insights.

### **Business Impact**

1.  **Cost Reduction**: Optimized inventory levels lead to reduced warehousing, holding, and spoilage/obsolescence costs. More accurate staff scheduling results in savings on unnecessary labor costs.
2.  **Revenue Increase**: The solution helps in the minimization of lost sales due to stockouts. It also aids in generating additional revenue by enhancing the effectiveness of promotions and timing them correctly.
3.  **Operational Efficiency & Strategic Decision-Making**: The model leads to improvement in supply chain processes and better resource allocation (financial, human resources). It also supports better planning of marketing campaigns, new product launches, and data-driven development of competitive strategies.

### **Dataset Features**

The primary dataset (`train.csv`) contains nearly a million observations across 1115 unique stores. Key features include:
-   **`Store`**: A unique identifier for each store, crucial for per-store analysis and joining with `store.csv` for store-specific attributes (like `StoreType`, `Assortment`, `CompetitionDistance`).
-   **`Sales`**: The target variable representing turnover for a given store on a specific day.
-   **`Customers`**: The number of customers, a likely strong predictor of sales.
-   **`Open`**: An indicator of whether the store was open (1) or closed (0); critical as closed stores have zero sales.
-   **`Promo`**: Indicates if a store was running a promotion, expected to significantly influence sales.
-   **`StateHoliday`**: Indicates type of state holiday ('a' = Public, 'b' = Easter, 'c' = Christmas, '0' = None), which usually has a major impact on sales.
-   **`SchoolHoliday`**: Indicates if sales were affected by public school closures, which can affect customer traffic.
-   **`Date`**: Transaction date, essential for time-series analysis and feature engineering (e.g., extracting month, year).

The supplementary dataset (`store.csv`) contains store-specific attributes such as `StoreType`, `Assortment` (basic, extra, extended), `CompetitionDistance`, `CompetitionOpenSinceMonth/Year`, `Promo2` (participation in a continuous promotion), and `PromoInterval`. These features provide context and are merged with `train.csv` to enhance model performance by accounting for store heterogeneity and competitive/promotional landscapes. These features contribute to the model's performance by allowing it to learn store-specific patterns, the impact of competition, and the effects of different types of promotions on sales.

### **Approach**

The modeling approach involves:
1.  **Data Preprocessing**: Converting the `Date` column to datetime objects. Encoding categorical features like `StateHoliday`, `StoreType`, and `Assortment`. Handling missing values in `store.csv`, particularly for `CompetitionDistance` and `CompetitionOpenSinceMonth/Year`, and addressing data quality issues like anomalous years for `CompetitionOpenSinceYear`.
2.  **Feature Engineering**: Creating features from date information (e.g., month, year, day of year). Deriving features like "duration of competition" from `CompetitionOpenSinceMonth/Year` and "duration of Promo2" from `Promo2SinceYear/Week`. Parsing `PromoInterval` to determine if Promo2 is active in a specific month.
3.  **Model Selection and Evaluation**: Evaluating various time-series and machine learning models such as ARIMA, Prophet, XGBoost, LightGBM, and Deep Learning models. The project notes that LightGBM with a base feature set of 26 features was a leading contender, achieving an RMSPE of approximately 11.90% and an MAE of about €527 on the full validation set. A Prophet model for a single store (Store 1) performed well in isolation (RMSPE ~9.19%) but was outperformed by the global LightGBM model for that specific store (LGBM RMSPE for Store 1: ~4.11%).
4.  **Success Metrics**: Key Performance Indicators (KPIs) used to measure success include RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error), and R² Score (Coefficient of Determination) for model performance.

The chosen methods (LightGBM, Prophet) are well-suited for time-series forecasting with complex interactions and seasonalities. LightGBM is a gradient boosting framework known for its efficiency and accuracy with large datasets and high-dimensional features. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects, which is robust to missing data and shifts in the trend and typically handles outliers well.

[Click here to explore the case study](https://github.com/edaaydinea/DataScienceForBusiness/blob/main/3.%20Sales%20Department%20Data/SalesDepartment.ipynb)