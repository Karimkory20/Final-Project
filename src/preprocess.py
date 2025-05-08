import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """Load and preprocess the cyberattack dataset."""
    print(f"Loading data from {file_path}...")
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert dates
        df['AttackDate'] = pd.to_datetime(df['AttackDate'])
        
        # Clean attack percentages
        df['Total_Attack_Percentage'] = df['Total_Attack_Percentage'].apply(
            lambda x: x.split('%')[0] if isinstance(x, str) else x
        )
        df['Total_Attack_Percentage'] = pd.to_numeric(df['Total_Attack_Percentage'], errors='coerce') / 100
        
        # Remove invalid data
        df = df.dropna(subset=['AttackDate', 'Country', 'Total_Attack_Percentage'])
        
        print(f"Data loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_top_10_countries(df):
    """Identify top 10 countries by average attack percentage."""
    print("Identifying top 10 countries by attack percentage...")
    
    # Calculate average attack percentage for each country
    country_avg = df.groupby('Country')['Total_Attack_Percentage'].agg([
        'mean',
        'count'
    ]).reset_index()
    
    # Filter countries with sufficient data (at least 30 days)
    country_avg = country_avg[country_avg['count'] >= 30]
    
    # Get top 10 countries
    top_10 = country_avg.nlargest(10, 'mean')['Country'].tolist()
    
    print("Top 10 countries identified:")
    for i, country in enumerate(top_10, 1):
        print(f"{i}. {country}")
    
    return top_10

def prepare_time_series(df, country):
    """Prepare time series data for a specific country."""
    # Filter data for the country
    country_df = df[df['Country'] == country].sort_values('AttackDate')
    
    # Create time series
    series = country_df[['AttackDate', 'Total_Attack_Percentage']].set_index('AttackDate')
    
    return series

def create_sequences(data, seq_length=10):
    """Create sequences for LSTM input."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])
    
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    # Test the preprocessing
    df = load_and_preprocess_data("Data/cleanedd_Attack_file.csv")
    if df is not None:
        top_10 = get_top_10_countries(df)
        print("\nTesting time series preparation for first country...")
        series = prepare_time_series(df, top_10[0])
        print(f"Time series shape: {series.shape}")