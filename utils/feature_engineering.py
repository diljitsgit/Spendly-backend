import pandas as pd
from datetime import datetime
import numpy as np

def engineer_features(df):
    """
    Engineer features from raw expense data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing raw expense data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Process date
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Extract temporal features
    data['day_of_week'] = data['date'].dt.day_name()
    data['month'] = data['date'].dt.month_name()
    data['day'] = data['date'].dt.day
    data['is_weekend'] = data['date'].dt.dayofweek >= 5
    
    # Amount-based features
    # Normalize amount by individual's spending patterns
    if 'age' in data.columns and 'gender' in data.columns:
        # Group by demographics for relative spending
        data['amount_float'] = pd.to_numeric(data['amount'], errors='coerce')
        
        # Calculate relative amount compared to average spending in same category
        category_avg = data.groupby('category')['amount_float'].transform('mean')
        data['relative_amount'] = data['amount_float'] / category_avg
        
        # Calculate relative amount compared to individual's average
        person_avg = data.groupby(['age', 'gender'])['amount_float'].transform('mean')
        data['personal_relative_amount'] = data['amount_float'] / person_avg
    
    # Category encoding with frequency
    category_counts = data['category'].value_counts(normalize=True)
    data['category_frequency'] = data['category'].map(category_counts)
    
    # Day of month patterns (beginning, middle, end)
    data['month_period'] = pd.cut(
        data['day'], 
        bins=[0, 10, 20, 32], 
        labels=['beginning', 'middle', 'end']
    )
    
    # One-hot encoding for categorical features
    categorical_features = ['category', 'day_of_week', 'month', 'gender', 'month_period']
    data_encoded = pd.get_dummies(
        data, 
        columns=categorical_features, 
        drop_first=False, 
        prefix=['cat', 'dow', 'month', 'gender', 'period']
    )
    
    # Remove original date and any text columns that would cause issues for the model
    if 'item' in data_encoded.columns:
        data_encoded = data_encoded.drop(columns=['item'])
    
    data_encoded = data_encoded.drop(columns=['date'])
    
    return data_encoded

def prepare_for_prediction(df):
    """
    Prepare new data for prediction, similar to training data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing new expense data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for prediction
    """
    # Engineer features
    data_prepared = engineer_features(df)
    
    # Convert any remaining non-numeric columns to numeric
    for col in data_prepared.columns:
        if data_prepared[col].dtype == 'object':
            try:
                data_prepared[col] = pd.to_numeric(data_prepared[col], errors='coerce')
            except:
                # Drop columns that can't be converted to numeric
                data_prepared = data_prepared.drop(columns=[col])
    
    return data_prepared