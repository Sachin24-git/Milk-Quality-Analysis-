import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_milk_quality_data(n_samples=1000):
    """
    Generate synthetic milk quality dataset for clustering analysis
    """
    np.random.seed(42)
    
    # Generate dates for the past year
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=x) for x in range(n_samples)]
    
    data = {
        'date': dates,
        'pH': np.random.normal(6.7, 0.2, n_samples),  # Ideal pH: 6.5-6.7
        'temperature': np.random.normal(4, 2, n_samples),  # Storage temp in °C
        'taste': np.random.randint(1, 11, n_samples),  # Taste score 1-10
        'odor': np.random.randint(1, 11, n_samples),   # Odor score 1-10
        'fat': np.random.normal(3.5, 0.5, n_samples),  # Fat content %
        'turbidity': np.random.normal(2.5, 1, n_samples),  # NTU units
        'color': np.random.normal(85, 10, n_samples),  # Whiteness score
        'grade': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    # Create some realistic patterns and correlations
    df = pd.DataFrame(data)
    
    # Higher fat usually means better taste
    df['taste'] = np.where(df['fat'] > 3.8, 
                          df['taste'] + np.random.randint(1, 3, n_samples), 
                          df['taste'])
    
    # Lower pH (spoilage) affects odor negatively
    df['odor'] = np.where(df['pH'] < 6.5, 
                         df['odor'] - np.random.randint(1, 4, n_samples), 
                         df['odor'])
    
    # Temperature affects quality
    df['taste'] = np.where(df['temperature'] > 6, 
                          df['taste'] - np.random.randint(1, 3, n_samples), 
                          df['taste'])
    
    # Ensure scores are within bounds
    df['taste'] = np.clip(df['taste'], 1, 10)
    df['odor'] = np.clip(df['odor'], 1, 10)
    
    # Add batch IDs
    df['batch_id'] = [f'BATCH_{i:04d}' for i in range(1, n_samples + 1)]
    
    # Reorder columns
    df = df[['batch_id', 'date', 'pH', 'temperature', 'taste', 'odor', 
             'fat', 'turbidity', 'color', 'grade']]
    
    return df

def save_dataset():
    """Generate and save the dataset"""
    df = generate_milk_quality_data(1500)
    df.to_csv('data/milk_quality.csv', index=False)
    print(f"Dataset generated with {len(df)} samples")
    print("Sample data:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())

if __name__ == "__main__":
    save_dataset()