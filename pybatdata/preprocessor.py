import numpy as np
import pandas as pd
import os
import json
import random
import re
import time

class Preprocessor:
    def __init__(self, folderpath, test_name):
        print("Preprocessor running...")
        self.folderpath = folderpath
        self.test_name = test_name
        self.record = self.read_record()
        self.column_headings = ['Date', 'Time', 'Cycle', 'Step', 'Current (A)',  'Voltage (V)', 'Capacity (Ah)', 'Discharge Capacity (Ah)', 'Charge Capacity (Ah)','dQ/dV (Ah/V)', 'Exp Capacity (Ah)', 'Cycle Capacity (Ah)']
        self.data_verified = False
        self.update_parquets = False
        for i in range(len(self.record)):
            print(f"Processing record {i+1} of {len(self.record)}:")
            test_details = self.write_metadata(i)
            self.write_parquet(test_details)
            
    def data_path(self, cell_name):
        return os.path.join(self.folderpath, f"{cell_name}", f'{self.test_name}.parquet')
    
    def metadata_path(self, cell_name):
        return os.path.join(self.folderpath, f"{cell_name}", f'{self.test_name}_details.json')
    
    def csv_path(self, cycler, channel):
        return os.path.join(self.folderpath, self.test_name, f"{self.test_name}_{cycler}_{channel}.csv")
        
    def read_record(self):
        record_xlsx = os.path.join(self.folderpath, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = self.test_name)
    
    def write_metadata(self, record_entry):
        test_details = self.record.iloc[record_entry].to_dict()
        cell_num = int(test_details['Cell'])
        cell_name = f"Cell_{cell_num}"
        

        # create directory for cell if it doesn't exist
        if not os.path.exists(os.path.dirname(self.metadata_path(cell_name))):
            os.makedirs(os.path.dirname(self.metadata_path(cell_name)))
        
        if os.path.exists(self.metadata_path(cell_name)):
            with open(self.metadata_path(cell_name), 'r') as f:
                existing_details = json.load(f)
                if existing_details != test_details:
                    with open(self.metadata_path(cell_name), 'w') as f:
                        json.dump(test_details, f)
        else:
            with open(self.metadata_path(cell_name), 'w') as f:
                json.dump(test_details, f)
        return test_details
                
    def write_parquet(self, test_details):
        cycler = int(test_details['Cycler']) if not pd.isna(test_details['Cycler']) else None
        channel = int(test_details['Channel']) if not pd.isna(test_details['Channel']) else None
        cell_num = int(test_details['Cell'])
        cell_name = f"Cell_{cell_num}"
        if os.path.exists(self.data_path(cell_name)):
            existing_data = pd.read_parquet(self.data_path(cell_name))
            # Compare headings
            existing_columns = list(existing_data.columns)
            if (self.column_headings != existing_columns) or (self.update_parquets==True):
                t1 = time.time()
                test_data = Neware.import_csv(self.csv_path(cycler, channel))
                test_data.to_parquet(self.data_path(cell_name), engine='pyarrow')
                print(f"\tparquet updated in {time.time()-t1:.2f} seconds.")
            else:
                if self.data_verified==False:
                    test_data = Neware.import_csv(self.csv_path(cycler, channel))
                    # Compare a randomly selected row
                    random_index = random.choice(existing_data.index.tolist())
                    if not existing_data.loc[random_index].equals(test_data.loc[random_index]):
                        print(f"\tparquet does not match raw data.")
                        t1 = time.time()
                        test_data.to_parquet(self.data_path(cell_name), engine='pyarrow')
                        self.update_parquets = True
                        print(f"\tparquet updated in {time.time()-t1:.2f} seconds.")
                    else:
                        print(f"\tparquet matches raw data.")
                    self.data_verified = True
                else:
                    print("\tparquet unchanged.")
        else:
            t1 = time.time()
            test_data = Neware.import_csv(self.csv_path(cycler, channel))
            test_data.to_parquet(self.data_path(cell_name), engine='pyarrow')
            print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

class Neware: 
    @classmethod
    def import_csv(cls,filepath):
        df = pd.read_csv(filepath)
        column_dict = {'Date': 'Date', 'Time': 'Time', 'Cycle Index': 'Cycle', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'DChg. Cap.(Ah)': 'Discharge Capacity (Ah)', 'Chg. Cap.(Ah)': 'Charge Capacity (Ah)','dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}
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
        df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()
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