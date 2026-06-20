import pandas as pd
import numpy as np
import os

def generate_retail_data(days=365, items=5):
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=days)
    data = []
    
    for item in range(1, items + 1):
        base_sales = np.random.randint(20, 100)
        for date in dates:
            # Simulate seasonality and noise
            seasonality = np.sin(date.dayofyear / 365 * 2 * np.pi) * 10
            noise = np.random.normal(0, 5)
            sales = max(0, int(base_sales + seasonality + noise))
            
            # Simulate lead time (days it takes for stock to arrive)
            lead_time = np.random.randint(2, 6)
            
            data.append([date, f"Product_{item}", sales, lead_time])
            
    df = pd.DataFrame(data, columns=["Date", "Product_ID", "Sales", "Lead_Time"])
    os.makedirs('data', exist_ok=True)
    df.to_csv("data/retail_sales.csv", index=False)
    print("Synthetic dataset created at data/retail_sales.csv")

if __name__ == "__main__":
    generate_retail_data()