#%%
import pandas as pd
import re

loc = "/Users/tom/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD Workspace/05 - Parallel SLP/DATA"

# Load the data
df = pd.read_csv(loc + "/SLP_Cell-Cell_Variation_R1_4_1.csv")

def convert_units(df):
    conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}

    for column in df.columns:
        match = re.search(r'\((.*?)\)', column)
        if match:
            unit = match.group(1)
            prefix = next((x for x in unit if not x.isupper()), None)
            if prefix in conversion_dict:
                df[column] = df[column] * conversion_dict[prefix]
                df.rename(columns={column: column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')}, inplace=True)

    return df

column_dict = {'Date': 'Date', 'Time': 'Time', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}
df = convert_units(df)
df = df[list(column_dict.keys())].rename(columns=column_dict)
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()
print(df.head())
# %%
