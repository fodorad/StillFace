import pandas as pd

input_file = 'data/ELTE-PPK_StillFace/metadata_database.xlsx'
output_file = 'data/ELTE-PPK_StillFace/missing_files_report.csv'

df = pd.read_excel(input_file, keep_default_na=True, na_values=[''])

columns_to_check = [
    ('mother', 'camera'),
    ('baby', 'camera'),
    ('window', 'camera'),
    ('door', 'camera'),
    ('polar_mother', 'HRV'),
    ('polar_baby', 'HRV'),
]

missing_entries = []

for idx, row in df.iterrows():
    for col, filetype in columns_to_check:
        if str(row[col]).strip().lower() == 'n':
            missing_entries.append({
                'ID': row['ID'],
                'session_date': row['Session_date_(YYYY-HH-DD)'],
                'session_hour': row['Session_hour_(HH:MM)'],
                'missing': f"{filetype} {{{col}}}"
            })

out_df = pd.DataFrame(missing_entries)
out_df.to_csv(output_file, index=False)
