import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load and clean the cyberattack dataset."""
    try:
        df = pd.read_csv(file_path)
        df['AttackDate'] = pd.to_datetime(df['AttackDate'])
        df = df.sort_values(['AttackDate', 'Country'])
        # Handle concatenated percentage strings
        df['Total_Attack_Percentage'] = df['Total_Attack_Percentage'].apply(
            lambda x: x.split('%')[0] if isinstance(x, str) else x
        )
        # Remove percentage signs and convert to numeric
        df['Total_Attack_Percentage'] = pd.to_numeric(df['Total_Attack_Percentage'], errors='coerce') / 100
        df = df.dropna(subset=['Total_Attack_Percentage', 'Country', 'AttackDate'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_time_series(df, country, target_col='Total_Attack_Percentage'):
    """Create a time series for a specific country."""
    country_df = df[df['Country'] == country].sort_values('AttackDate')
    return country_df[['AttackDate', target_col]].set_index('AttackDate')

def preprocess_data(series, seq_length=10):
    """Convert time series into sequences for LSTM input."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    # Load data
    df = load_data(r"D:\DEPI DataAnalytics\Projects\cyberattack_analysis\Data\formatted_cyber_with_region.csv")
    if df is None:
        exit(1)

    # Identify top 5 countries by average attack percentage
    top_countries = df.groupby('Country')['Total_Attack_Percentage'].mean().nlargest(5).index

    # Process each country
    for country in top_countries:
        series = create_time_series(df, country)
        if not series.empty:
            X, y, scaler = preprocess_data(series)
            print(f"Processed {country}: X.shape={X.shape}, y.shape={y.shape}")
        else:
            print(f"No data for {country}")