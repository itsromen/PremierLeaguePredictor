import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Premier_League.csv')

# 1. Clean missing data
# First handle string 'Nan' values in attendance column
df['attendance'] = df['attendance'].replace(['Nan', 'NaN', 'nan'], np.nan)

# Drop rows with missing values in critical columns
df = df.dropna(subset=['stadium', 'attendance'])

# Clean attendance column - remove commas and convert to int
df['attendance'] = df['attendance'].astype(str).str.replace(',', '').astype(int)

# 2. Create target variable (match outcome)
conditions = [
    df['Goals Home'] > df['Away Goals'],
    df['Goals Home'] < df['Away Goals'],
    df['Goals Home'] == df['Away Goals']
]
choices = ['Home Win', 'Away Win', 'Draw']
df['outcome'] = np.select(conditions, choices)

# 3. Feature engineering
df['possession_difference'] = df['home_possessions'] - df['away_possessions']
df['shot_difference'] = df['home_shots'] - df['away_shots']

# 4. Select final features
final_columns = [
    'possession_difference',
    'shot_difference',
    'attendance',
    'outcome'
]
df = df[final_columns]

# 5. Save cleaned data
df.to_csv('premier_league_cleaned.csv', index=False)

print(f"Dataset cleaned successfully!")
print(f"Final dataset shape: {df.shape}")
print(f"Outcome distribution:")
print(df['outcome'].value_counts())