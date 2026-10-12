import pandas as pd
import os

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans the Online Retail dataset for customer segmentation.
    """
    print(f"Loading dataset from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path, encoding='latin1')
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows:,}")
    
    # 1. Drop rows missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # 2. Filter out cancelled orders (Quantity < 0 or InvoiceNo starts with 'C')
    df = df[df['Quantity'] > 0]
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    
    # 3. Remove outlier prices, non-product entries, and invalid rows
    df = df[df['UnitPrice'] > 0]
    non_product_codes = ['POST', 'DOT', 'M', 'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'PADS', 'DCGSSBOY', 'DCGSSGIRL']
    df = df[~df['StockCode'].isin(non_product_codes)]
    
    # Remove extreme price outliers using 99.9th percentile safeguard
    price_cap = df['UnitPrice'].quantile(0.999)
    df = df[df['UnitPrice'] <= price_cap]
    
    # 4. Feature Creation: Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Ensure InvoiceDate is parsed as datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    final_rows = len(df)
    print(f"Data cleaning complete. Retained {final_rows:,} rows ({initial_rows - final_rows:,} rows removed).")
    
    return df

if __name__ == "__main__":
    input_file = "data/online_retail.csv"
    output_file = "data/cleaned_online_retail.csv"
    
    os.makedirs("data", exist_ok=True)
    cleaned_df = load_and_clean_data(input_file)
    cleaned_df.to_csv(output_file, index=False)
    print(f"Cleaned dataset successfully saved to '{output_file}'.")