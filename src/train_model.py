import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_forecasting_model():
    df = pd.read_csv("data/processed_sales.csv")
    
    # Features and Target
    features = ['DayOfWeek', 'Month', 'Sales_Lag_1', 'Sales_Lag_7', 'Rolling_Mean_7']
    X = df[features]
    y = df['Sales']
    
    # Train-test split (chronological)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained. Mean Absolute Error: {mae:.2f} units")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_sales_model.pkl')
    print("Model saved to models/rf_sales_model.pkl")

if __name__ == "__main__":
    train_forecasting_model()