import pandas as pd
from datetime import datetime, timedelta

# Define the date range
start_date = datetime(2022, 10, 11)
end_date = datetime(2023, 12, 31)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Generate data for China
data = []
percentage = 0.025  # Starting value
for date in date_range:
    data.append({
        'AttackDate': date.strftime('%Y-%m-%d'),
        'Country': 'China',
        'Total_Attack_Percentage': percentage,
        'Region': 'Asia'
    })
    percentage += 0.00002  # Increment slightly each day

# Convert to DataFrame
df = pd.DataFrame(data)

# Append to the existing CSV
existing_df = pd.read_csv('data/formatted_cyber_with_region.csv')
updated_df = pd.concat([existing_df, df], ignore_index=True)
updated_df.to_csv('data/formatted_cyber_with_region.csv', index=False)
print("Added data for China to formatted_cyber_with_region.csv")