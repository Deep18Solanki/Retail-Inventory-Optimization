import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Product_ID', 'Date'])
    
    # Feature Engineering
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    
    # Lag features (past sales)
    df['Sales_Lag_1'] = df.groupby('Product_ID')['Sales'].shift(1)
    df['Sales_Lag_7'] = df.groupby('Product_ID')['Sales'].shift(7)
    
    # Rolling average
    df['Rolling_Mean_7'] = df.groupby('Product_ID')['Sales'].transform(lambda x: x.rolling(window=7).mean())
    
    df = df.dropna() # Drop rows with NaN due to shifting
    return df

if __name__ == "__main__":
    df = load_and_preprocess("data/retail_sales.csv")
    df.to_csv("data/processed_sales.csv", index=False)
    print("Data preprocessed successfully.")