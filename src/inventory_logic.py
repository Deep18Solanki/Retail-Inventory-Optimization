import pandas as pd
import numpy as np

def calculate_inventory_kpis(df, predictions):
    # Formulas:
    # Safety Stock = Z_score * Standard_Deviation_of_Demand * sqrt(Lead_Time)
    # Reorder Point = (Average_Daily_Demand * Lead_Time) + Safety_Stock
    
    z_score = 1.65 # 95% service level
    
    results = []
    products = df['Product_ID'].unique()
    
    for idx, prod in enumerate(products):
        prod_data = df[df['Product_ID'] == prod]
        std_demand = prod_data['Sales'].std()
        avg_lead_time = prod_data['Lead_Time'].mean()
        
        # Using a simplistic mock prediction for the next 7 days average
        avg_daily_demand_forecast = predictions[idx] if idx < len(predictions) else prod_data['Sales'].mean()
        
        safety_stock = z_score * std_demand * np.sqrt(avg_lead_time)
        reorder_point = (avg_daily_demand_forecast * avg_lead_time) + safety_stock
        
        results.append({
            "Product_ID": prod,
            "Forecasted_Daily_Demand": round(avg_daily_demand_forecast, 2),
            "Safety_Stock": round(safety_stock, 0),
            "Reorder_Point": round(reorder_point, 0)
        })
        
    return pd.DataFrame(results)