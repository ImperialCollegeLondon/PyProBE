import numpy as np
import pandas as pd
import os
import re
import time

class Preprocessor:
    def __init__(self, folderpath, test_name):
        print("Preprocessor running...")
        self.folderpath = folderpath
        self.test_name = test_name
        self.record = self.read_record()
        self.column_headings = ['Date', 
                                'Time', 
                                'Cycle', 
                                'Step', 
                                'Current (A)',  
                                'Voltage (V)', 
                                'Capacity (Ah)', 
                                'Discharge Capacity (Ah)', 
                                'Charge Capacity (Ah)',]
        self.test_dict = [None]*len(self.record)
        for record_entry in range(len(self.record)):
            print(f"Processing record {record_entry+1} of {len(self.record)}:")
            self.test_dict[record_entry] = self.record.iloc[record_entry].to_dict()
            self.write_parquet(self.test_dict[record_entry])
            

    def read_record(self):
        record_xlsx = os.path.join(self.folderpath, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = self.test_name)
            
    def xlsx_path(self, cycler, channel):
        return os.path.join(self.folderpath, self.test_name, f"{self.test_name}-{cycler}-{channel}.xlsx")
    
    def parquet_path(self, cycler, channel):
        xlsx_path = self.xlsx_path(cycler, channel)
        return xlsx_path.replace('.xlsx', '.parquet')
                
    def write_parquet(self, test_dict):
        cycler = int(test_dict['Cycler']) if not pd.isna(test_dict['Cycler']) else None
        channel = int(test_dict['Channel']) if not pd.isna(test_dict['Channel']) else None
        cell_num = int(test_dict['Cell'])
        cell_name = f"Cell_{cell_num}"
        if not os.path.exists(self.parquet_path(cycler, channel)):
            t1 = time.time()
            test_data = Neware.import_xlsx(self.xlsx_path(cycler, channel))
            test_data.to_parquet(self.parquet_path(cycler, channel), engine='pyarrow')
            print(f"\tparquet written in {time.time()-t1:.2f} seconds.")
        else:
            print("\tparquet already exists.")

class Neware: 
    @staticmethod
    def import_xlsx(cls,filepath):
        df = pd.read_excel(filepath)
        column_dict = {'Date': 'Date', 
                       'Cycle Index': 'Cycle', 
                       'Step Index': 'Step', 
                       'Current(A)': 'Current (A)', 
                       'Voltage(V)': 'Voltage (V)', 
                       'DChg. Cap.(Ah)': 'Discharge Capacity (Ah)', 
                       'Chg. Cap.(Ah)': 'Charge Capacity (Ah)',
                       }
        df = cls.convert_units(df)
        df = df[list(column_dict.keys())].rename(columns=column_dict)
        dQ_charge = np.diff(df['Charge Capacity (Ah)'])
        dQ_discharge = np.diff(df['Discharge Capacity (Ah)'])
        dQ_charge[dQ_charge < 0] = 0
        dQ_discharge[dQ_discharge < 0] = 0
        dQ_charge = np.append(dQ_charge, 0)
        dQ_discharge = np.append(dQ_discharge, 0)
        df['Capacity (Ah)'] = np.cumsum(dQ_charge - dQ_discharge)
        df['Capacity (Ah)'] = df['Capacity (Ah)'] + df['Charge Capacity (Ah)'].max()
        df['Exp Capacity (Ah)'] = np.zeros(len(df))
        df['Cycle Capacity (Ah)'] = np.zeros(len(df))
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
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