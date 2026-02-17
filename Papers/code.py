import pandas as pd

# Define possible column name variations for each required field
column_map = {
    'participant_name': ["Player's Name", 'name of participant', 'participant', 'founder name', 'name'],
    'company_name': ['Q1: Company Name', 'name of company', 'company', 'startup name', 'company name'],
    'company_description': ['Q2: Describe your company in 50 characters', 'description of company', 'company description', 'about company', 'description'],
    'linkedin_profile': ['Q6: LinkedIn Profiles', 'linkedin profile', 'linkedin', 'linkedin url', 'linkedin link'],
    'pitch_deck': ['Q21: Attach your business deck/ pitch/ demo video', 'pitch deck', 'pitchdeck', 'pitch deck link', 'deck']
}

def find_column(possible_names, columns):
    for name in possible_names:
        for col in columns:
            if name.strip().lower() in col.strip().lower():
                return col
    return None

# Read the original CSV file
df = pd.read_csv('Startups_IGNITE.csv')

# Find actual column names in the file
selected_columns = {}
for key, possible_names in column_map.items():
    col = find_column(possible_names, df.columns)
    if col:
        selected_columns[key] = col
    else:
        print(f"Warning: Could not find column for {key}")

# Select only the found columns
output_columns = [col for col in selected_columns.values()]
cleaned_df = df[output_columns]

# Rename columns to standard names
rename_map = {v: k for k, v in selected_columns.items()}
cleaned_df = cleaned_df.rename(columns=rename_map)

# Save to new Excel file
cleaned_df.to_excel('Cleaned_Startups_IGNITE.xlsx', index=False)