import pandas as pd
import numpy as np
import os

def compute_rfm_segments(cleaned_file_path: str) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary (RFM) values, scores them using quartiles,
    and assigns behavioral customer segments.
    """
    print(f"Loading cleaned dataset from {cleaned_file_path}...")
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_file_path}")
        
    df = pd.read_csv(cleaned_file_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Define snapshot date (1 day after the last transaction)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Snapshot date set to: {snapshot_date.date()}")
    
    print("Aggregating transactional data to customer level (RFM)...")
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                                  # Frequency
        'TotalAmount': 'sum'                                     # Monetary
    }).reset_index()
    
    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm_df = rfm_df[(rfm_df['Monetary'] > 0) & (rfm_df['Frequency'] > 0)]
    
    print("Assigning RFM Quartile Scores...")
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=4, labels=r_labels, duplicates='drop')
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=f_labels)
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=4, labels=m_labels)
    
    rfm_df['R_Score'] = rfm_df['R_Score'].astype(int)
    rfm_df['F_Score'] = rfm_df['F_Score'].astype(int)
    rfm_df['M_Score'] = rfm_df['M_Score'].astype(int)
    
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
    
    print("Mapping segments to customer personas...")
    def segment_customer(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 3 and f >= 3 and m >= 3:
            return 'Champions'
        elif f >= 3 and m >= 2:
            return 'Loyal Customers'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        elif r >= 4 and f == 1:
            return 'Recent Customers'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r == 1 and f >= 3 and m >= 3:
            return 'Cannot Lose Them'
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        else:
            return 'Needs Attention'

    rfm_df['Customer_Segment'] = rfm_df.apply(segment_customer, axis=1)
    
    print("\nSegment Distribution Summary:")
    print(rfm_df['Customer_Segment'].value_counts())
    
    return rfm_df

if __name__ == "__main__":
    input_path = "data/cleaned_online_retail.csv"
    output_path = "data/customer_rfm_segments.csv"
    
    rfm_result = compute_rfm_segments(input_path)
    rfm_result.to_csv(output_path, index=False)
    print(f"\nRFM segmentation dataset successfully saved to '{output_path}'.")