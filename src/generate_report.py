import pandas as pd
import json

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
    for country in top_countries:
        series = df[df['Country'] == country][['AttackDate', 'Total_Attack_Percentage']]
        if series.empty:
            print(f"No data for {country} after filtering.")
            continue
        # Scale the Total_Attack_Percentage for better visualization
        series['Total_Attack_Percentage'] = series['Total_Attack_Percentage'] * 100
        series['AttackDate'] = series['AttackDate'].dt.strftime('%Y-%m-%d')
        country_data.append({
            'name': country,
            'data': series.to_dict(orient='records')
        })
        print(f"Processed data for {country} with {len(series)} records.")

    # Average attack percentage by region
    regions = df[df['Region'] != 'Unknown Region']['Region'].unique()
    print("Regions in dataset:", regions)
    region_data = []
    for region in regions:
        avg_attack = df[df['Region'] == region]['Total_Attack_Percentage'].mean() * 100  # Scale for consistency
        region_data.append({'region': region, 'avgAttack': float(avg_attack)})
        print(f"Average attack percentage for {region}: {avg_attack:.2f}%")

    # Save to JSON
    output_data = {'countries': country_data, 'regions': region_data}
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Generated {output_file} with {len(country_data)} countries and {len(region_data)} regions.")

if __name__ == "__main__":
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(r"D:\DEPI DataAnalytics\Projects\cyberattack_analysis\Data\formatted_cyber_with_region.csv")
    generate_json_data(df)