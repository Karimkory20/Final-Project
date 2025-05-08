import pandas as pd
import numpy as np
from preprocess import load_and_preprocess_data, get_top_10_countries
from lstm import train_lstm_model, generate_predictions
from generate_report import generate_json_report
import os

def main():
    print("Starting Cyberattack Analysis Pipeline...")
    
    # Step 1: Preprocess data for top 10 dangerous countries
    print("\n1. Preprocessing data for top 10 dangerous countries...")
    df = load_and_preprocess_data("Data/cleanedd_Attack_file.csv")
    top_10_countries = get_top_10_countries(df)
    print(f"Top 10 countries identified: {', '.join(top_10_countries)}")
    
    # Step 2: Train LSTM models and generate predictions
    print("\n2. Training LSTM models and generating predictions...")
    predictions = {}
    for country in top_10_countries:
        print(f"\nProcessing {country}...")
        model, scaler = train_lstm_model(df, country)
        future_predictions = generate_predictions(model, scaler, df, country)
        predictions[country] = future_predictions
        print(f"Generated 6-month predictions for {country}")
    
    # Step 3: Generate report
    print("\n3. Generating report...")
    report_data = generate_json_report(df, predictions, top_10_countries)
    
    # Step 4: Ensure web directory exists
    os.makedirs('web', exist_ok=True)
    
    # Save the report
    with open('web/data.json', 'w') as f:
        import json
        json.dump(report_data, f, indent=2)
    
    print("\nPipeline completed successfully!")
    print("Report generated at: web/data.json")
    print("Open web/report.html to view the visualization")

if __name__ == "__main__":
    main() 