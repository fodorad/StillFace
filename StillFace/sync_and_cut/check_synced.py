import pandas as pd

input_file = 'data/ELTE-PPK_StillFace/metadata_database.xlsx'

df = pd.read_excel(input_file, keep_default_na=True, na_values=[''])