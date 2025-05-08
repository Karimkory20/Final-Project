import pandas as pd
import json
from lstm import load_data, create_time_series, preprocess_data, train_model, predict_future
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os

def generate_json_data(df, output_file='web/data.json'):
    """Generate JSON data for the HTML report."""
    print("Starting data processing...")
    
    # Ensure AttackDate is datetime
    df['AttackDate'] = pd.to_datetime(df['AttackDate'])
    df = df.dropna(subset=['AttackDate', 'Country', 'Total_Attack_Percentage', 'Region'])
    print(f"Dataset loaded with {len(df)} rows after dropping NaN values.")

    # Convert Total_Attack_Percentage to numeric after removing '%' sign
    df['Total_Attack_Percentage'] = df['Total_Attack_Percentage'].str.rstrip('%').astype(float) / 100

    # Log unique country names to help debug missing "People's Republic of China"
    unique_countries = df['Country'].unique()
    print("Countries in dataset:", unique_countries)

    # Time series data for top 4 countries + People's Republic of China (or China)
    top_countries = df.groupby('Country')['Total_Attack_Percentage'].mean().nlargest(4).index.tolist()
    # Check for both possible names
    china_name = None
    if "People's Republic of China" in unique_countries:
        china_name = "People's Republic of China"
    elif "China" in unique_countries:
        china_name = "China"
    
    if china_name and china_name not in top_countries:
        top_countries.append(china_name)
        print(f"Added {china_name} to the list of countries.")
    else:
        warning_message = china_name or "People's Republic of China/China"
        print(f"Warning: {warning_message} not found in dataset.")

    country_data = []
    future_predictions = {}
    
    # Process each country with LSTM predictions
    for country in top_countries:
        print(f"\nProcessing LSTM predictions for {country}...")
        series = create_time_series(df, country)
        if series.empty:
            print(f"No data for {country} after filtering.")
            continue
            
        # Generate LSTM predictions
        X, y, scaler = preprocess_data(series)
        if len(X) < 20:
            print(f"Insufficient data for LSTM predictions for {country}")
            continue
            
        X_flat = X.reshape((X.shape[0], X.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        last_sequence = X[-1].reshape(-1)
        future_vals = predict_future(model, last_sequence, scaler, steps=180)
        
        # Calculate percentage change
        avg_future = np.mean(future_vals)
        last_value = scaler.inverse_transform([y[-1]])[0]
        percentage_change = ((avg_future - last_value) / last_value) * 100
        
        # Store historical data and predictions
        series['Total_Attack_Percentage'] = series['Total_Attack_Percentage'] * 100
        series['AttackDate'] = series['AttackDate'].dt.strftime('%Y-%m-%d')
        
        # Generate future dates for predictions
        last_date = pd.to_datetime(series.index[-1])
        future_dates = pd.date_range(start=last_date, periods=181, freq='D')[1:]
        future_series = pd.Series(future_vals, index=future_dates)
        future_series.index = future_series.index.strftime('%Y-%m-%d')
        
        country_data.append({
            'name': country,
            'data': series.to_dict(orient='records'),
            'predictions': future_series.to_dict(),
            'prediction_change': float(percentage_change)
        })
        print(f"Processed data for {country} with {len(series)} records and future predictions.")

    # Average attack percentage by region
    regions = df[df['Region'] != 'Unknown Region']['Region'].unique()
    print("Regions in dataset:", regions)
    region_data = []
    for region in regions:
        avg_attack = df[df['Region'] == region]['Total_Attack_Percentage'].mean() * 100  # Scale for consistency
        region_data.append({'region': region, 'avgAttack': float(avg_attack)})
        print(f"Average attack percentage for {region}: {avg_attack:.2f}%")

    # Save to JSON
    output_data = {
        'countries': country_data,
        'regions': region_data,
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Generated {output_file} with {len(country_data)} countries and {len(region_data)} regions.")

def generate_json_report(df, predictions, top_10_countries):
    """Generate JSON report for the top 10 countries."""
    print("Generating JSON report...")
    
    # Prepare country data
    country_data = []
    for country in top_10_countries:
        # Get historical data
        country_df = df[df['Country'] == country].sort_values('AttackDate')
        historical_data = country_df[['AttackDate', 'Total_Attack_Percentage']].to_dict('records')
        
        # Convert dates to strings
        for record in historical_data:
            record['AttackDate'] = record['AttackDate'].strftime('%Y-%m-%d')
            record['Total_Attack_Percentage'] = float(record['Total_Attack_Percentage'] * 100)
        
        # Get predictions
        country_predictions = predictions[country]
        
        # Generate future dates
        last_date = pd.to_datetime(historical_data[-1]['AttackDate'])
        future_dates = [last_date + timedelta(days=i+1) for i in range(len(country_predictions))]
        
        # Create prediction records
        prediction_records = {
            date.strftime('%Y-%m-%d'): float(pred * 100)
            for date, pred in zip(future_dates, country_predictions)
        }
        
        # Calculate prediction change
        last_value = historical_data[-1]['Total_Attack_Percentage']
        avg_future = sum(prediction_records.values()) / len(prediction_records)
        prediction_change = ((avg_future - last_value) / last_value) * 100
        
        # Add to country data
        country_data.append({
            'name': country,
            'data': historical_data,
            'predictions': prediction_records,
            'prediction_change': float(prediction_change)
        })
    
    # Calculate regional averages
    regions = df[df['Region'] != 'Unknown Region']['Region'].unique()
    region_data = []
    for region in regions:
        avg_attack = df[df['Region'] == region]['Total_Attack_Percentage'].mean() * 100
        region_data.append({
            'region': region,
            'avgAttack': float(avg_attack)
        })
    
    # Create final report
    report_data = {
        'countries': country_data,
        'regions': region_data,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("Report generation completed")
    return report_data

if __name__ == "__main__":
    # Test the report generation
    from preprocess import load_and_preprocess_data, get_top_10_countries
    from lstm import train_lstm_model, generate_predictions
    
    print("Starting report generation for top 10 countries...")
    
    # Load and preprocess data
    df = load_and_preprocess_data("Data/cleanedd_Attack_file.csv")
    if df is not None:
        # Get top 10 countries
        top_10 = get_top_10_countries(df)
        print(f"\nProcessing predictions for {len(top_10)} countries...")
        
        # Generate predictions for all countries
        predictions = {}
        for country in top_10:
            print(f"\nProcessing {country}...")
            try:
                model, scaler = train_lstm_model(df, country)
                country_predictions = generate_predictions(model, scaler, df, country)
                predictions[country] = country_predictions
                print(f"Generated predictions for {country}")
            except Exception as e:
                print(f"Error processing {country}: {str(e)}")
                continue
        
        # Generate complete report
        print("\nGenerating final report...")
        report_data = generate_json_report(df, predictions, top_10)
        
        # Save report
        os.makedirs('web', exist_ok=True)
        with open('web/data.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print("\nReport Summary:")
        print(f"Total countries processed: {len(report_data['countries'])}")
        print(f"Regions analyzed: {len(report_data['regions'])}")
        print(f"Report generated at: {report_data['last_updated']}")
        print("\nReport saved to web/data.json")
        
        # Print country-specific summary
        print("\nCountry Analysis Summary:")
        for country_data in report_data['countries']:
            country = country_data['name']
            historical_avg = sum(record['Total_Attack_Percentage'] for record in country_data['data']) / len(country_data['data'])
            prediction_avg = sum(country_data['predictions'].values()) / len(country_data['predictions'])
            change = country_data['prediction_change']
            
            print(f"\n{country}:")
            print(f"  Historical Average: {historical_avg:.2f}%")
            print(f"  Predicted Average: {prediction_avg:.2f}%")
            print(f"  Expected Change: {change:+.2f}%")