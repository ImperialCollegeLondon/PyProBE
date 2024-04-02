import pandas as pd
import numpy as np
import os
import re

class Neware: 
    @classmethod
    def load_file(cls, filepath):
        file_ext = os.path.splitext(filepath)[1]
        if file_ext == '.xlsx':
            df = pd.read_excel(filepath)
        elif file_ext == '.csv':
            df = pd.read_csv(filepath)
        column_dict = {'Date': 'Date', 
                       'Time': 'Step Time (s)',
                       'Cycle Index': 'Cycle', 
                       'Step Index': 'Step', 
                       'Current(A)': 'Current (A)', 
                       'Voltage(V)': 'Voltage (V)', 
                       'DChg. Cap.(Ah)': 'Step Discharge Capacity (Ah)', 
                       'Chg. Cap.(Ah)': 'Step Charge Capacity (Ah)',
                       }
        df = cls.convert_units(df)
        df = df[list(column_dict.keys())].rename(columns=column_dict)
        dQ_charge = np.diff(df['Charge Capacity (Ah)'])
        dQ_discharge = np.diff(df['Discharge Capacity (Ah)'])
        dQ_charge[dQ_charge < 0] = 0
        dQ_discharge[dQ_discharge < 0] = 0
        dQ_charge = np.append(0, dQ_charge)
        dQ_discharge = np.append(0, dQ_discharge)
        df['Capacity (Ah)'] = np.cumsum(dQ_charge - dQ_discharge)
        df['Capacity (Ah)'] = df['Capacity (Ah)'] + df['Charge Capacity (Ah)'].max()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time (s)'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
        df = df[['Date',
                 'Time (s)',
                 'Cycle',
                 'Step',
                 'Current (A)',
                 'Voltage (V)',
                 'Capacity (Ah)',
                 ]]
        return df
    
    @staticmethod
    def convert_units(df):
        conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}
        for column in df.columns:
            match = re.search(r'\((.*?)\)', column)  # Update the regular expression pattern to extract the unit from the column name
            if match:
                unit = match.group(1)
                prefix = next((x for x in unit if not x.isupper()), None)
                if prefix in conversion_dict:
                    df[column] = df[column] * conversion_dict[prefix]
                    df.rename(columns={column: column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')}, inplace=True)  # Update the logic for replacing the unit in the column name

        return df