🛒 E-Commerce Customer Segmentation & Lifetime Value (LTV) Prediction📌 

Project Overview
In competitive retail landscapes, treating all customers uniformly results in wasted marketing spend and high churn rates. This project delivers an end-to-end data analytics and predictive modeling solution designed to help e-commerce businesses identify high-value customer segments and forecast future revenue.Using transactional data from the UCI Online Retail repository, this pipeline cleans raw logs, executes RFM (Recency, Frequency, Monetary) behavioral segmentation, and deploys a Random Forest machine learning model to predict Customer Lifetime Value (LTV).🏗️ Project Architecture & Repository 

StructurePlaintextecommerce-customer-segmentation/
│
├── data/                      # Raw and processed datasets (gitignored)
│   ├── online_retail.csv
│   ├── cleaned_online_retail.csv
│   ├── customer_rfm_segments.csv
│   └── customer_ltv_predictions.csv
│
├── notebooks/                 # Exploratory data analysis & modeling notebooks
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_rfm_segmentation.ipynb
│   └── 03_ltv_prediction_modeling.ipynb
│
├── src/                       # Modular Python scripts
│   ├── data_cleaning.py
│   ├── rfm_segmentation.py
│   └── ltv_modeling.py
│
├── dashboard/                 # Power BI template & Streamlit app code
│   ├── app.py                 # Streamlit web app interface
│   └── ecom_dashboard.pbix    # Power BI executive report
│
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies

🔍 Methodology & Pipeline 
Steps1. Data Extraction & Defensive Cleaning (src/data_cleaning.py)Handled raw transactional inputs, dropping rows missing critical identifiers (CustomerID).Filtered out cancelled orders (Quantity < 0 and invoice numbers starting with 'C').Removed administrative and non-product stock codes (e.g., POST, DOT, BANK CHARGES) to prevent distortion of financial metrics.Engineered base feature: TotalAmount = Quantity * UnitPrice.2. Behavioral Segmentation via RFM (src/rfm_segmentation.py)Evaluated customer behavior relative to a fixed snapshot date:Recency (R): Days since last purchase.Frequency (F): Total unique orders placed.Monetary (M): Total historical revenue generated.Assigned quartile scores (1–4) and mapped customers into actionable personas such as Champions, Loyal Customers, At-Risk, and Hibernating.3. Predictive LTV Modeling (src/ltv_modeling.py)Implemented a two-period validation framework:Observation Window: First 9 months used to extract historical behavioral features.Prediction Window: Subsequent 3 months used to calculate actual future spend (Target $Y$).Trained a Random Forest Regressor using behavioral features (Recency, Frequency, Monetary_Total, Monetary_Mean, Monetary_Max).Evaluated model performance using MAE, RMSE, and $R^2$ Score.📊 Key Findings & Business InsightsThe Pareto Principle in Action: Approximately 18% of customers categorized as Champions and Loyal Customers account for over 65% of total revenue.At-Risk Interventions: A significant cluster of high-monetary customers have high recency gaps (>90 days since last purchase), representing an immediate target group for automated win-back email campaigns.Predictive Drivers: Feature importance analysis revealed that historical Total Monetary Spend and Purchase Frequency are the strongest predictors of future 3-month LTV, outperforming recency alone.🚀 Dashboards & Interactive AppPower BI Dashboard: An executive-ready visual report tracks revenue trends, RFM segment distribution sizes, and geographic performance. (Preview: Insert screenshot or link to hosted Power BI report here)Streamlit Web App: An interactive application where users can inspect customer segments and simulate LTV predictions based on user inputs.⚙️ How to Run LocallyClone the repository:Bashgit clone https://github.com/your-username/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation
Install dependencies:Bashpip install -r requirements.txt
Run the data cleaning and modeling pipeline:Bashpython src/data_cleaning.py
python src/rfm_segmentation.py
python src/ltv_modeling.py
Launch the Streamlit App:Bashstreamlit run dashboard/app.py
👤 Author: Desire Kofi Korda
LinkedIn Profile:
Portfolio Website: 