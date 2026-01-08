import pandas as pd
import os

# Ensure this matches your file name exactly
source_file = r'C:\Users\shach\OneDrive\שולחן העבודה\Miki London Lab\Mice learning analysis\mice_water_consumption_report.xlsx'

print(f"Reading {source_file}...")

# 1. Load the data using the openpyxl engine (required for .xlsx)
try:
    df = pd.read_excel(source_file, engine='openpyxl')
except Exception as e:
    print(f"Excel read failed ({e}), attempting CSV read...")
    df = pd.read_csv(source_file)


# 2. Define the Robust Date Parser
def parse_mixed_dates(val):
    # Convert to string and remove time " 00:00:00" if present
    s = str(val).strip().split(' ')[0]

    # Logic A: Slashed Format (DD/MM/YYYY)
    if '/' in s:
        return pd.to_datetime(s, dayfirst=True)

    # Logic B: Dashed Format
    # Try the "Bad Export" format (YYYY-DD-MM) first
    try:
        return pd.to_datetime(s, format='%Y-%d-%m')
    except:
        # Fallback to standard format if that fails (e.g. Month 13)
        return pd.to_datetime(s, errors='coerce')


# 3. Apply Parser and Sort
print("Parsing and sorting dates...")
df['clean_datetime'] = df['date'].apply(parse_mixed_dates)
df = df.sort_values('clean_datetime')

# 4. Format Output as DD-MM-YYYY
df['formatted_date'] = df['clean_datetime'].dt.strftime('%d-%m-%Y')

# 5. Save the Cleaned File
output_file = 'mice_water_consumption_cleaned.csv'

# Select columns to save (Formatted Date + Data Columns)
cols_to_save = ['formatted_date'] + [c for c in df.columns if c in ['P1L', 'P1R', 'PNO']]
df[cols_to_save].to_csv(output_file, index=False)

print(f"\nSuccess! File saved to: {os.path.abspath(output_file)}")
print("First 5 rows:")
print(df[cols_to_save].head())