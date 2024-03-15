import os
import polars as pl
import re

class Import_data:
    def __init__(self, folderpath, test_name):
        self.folderpath = folderpath
        self.test_name = test_name
        self.record = self.read_record()
        self.data = Neware.import_csv(os.path.join(self.folderpath, f"{self.test_name}/SLP_Cell-Cell_Variation_R1_4_1.csv"))
        
    def read_record(self):
        record_xlsx = os.path.join(self.folderpath, "Experiment_Record.xlsx")
        return pl.read_excel(record_xlsx, sheet_name = self.test_name)
    
    
class Neware: 
    @classmethod
    def import_csv(cls, filepath):

        # Read the CSV file
        df = pl.scan_csv(filepath, ignore_errors=True)
        print(df)
        # Define the new column names
        column_dict = {'Date': 'Date', 'Time': 'Time', 'Cycle Index': 'Cycle', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'DChg. Cap.(Ah)': 'Discharge Capacity (Ah)', 'Chg. Cap.(Ah)': 'Charge Capacity (Ah)','dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}

        new_df = cls.convert_units(df)
        print(new_df.collect().head())
        # Rename the columns
        #df = df.rename(column_dict)
        #print(df.columns)
        # for column, dtype in schema.items():
        #     if column in df.columns:
        #         df = df.with_column(column, pl.col(column).cast(dtype))

        # column_dict = {'Date': 'Date', 'Time': 'Time', 'Cycle Index': 'Cycle', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'DChg. Cap.(Ah)': 'Discharge Capacity (Ah)', 'Chg. Cap.(Ah)': 'Charge Capacity (Ah)','dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}
        # print(df.collect())
        # df = cls.convert_units(df)
        
        # df = df.select(list(column_dict.keys())).rename(column_dict)
        # dQ_charge = df['Charge Capacity (Ah)'].diff()
        # dQ_discharge = df['Discharge Capacity (Ah)'].diff()
        # dQ_charge = dQ_charge.where(dQ_charge >= 0, 0)
        # dQ_discharge = dQ_discharge.where(dQ_discharge >= 0, 0)
        # dQ_charge = dQ_charge.append(pl.Series([0]))
        # dQ_discharge = dQ_discharge.append(pl.Series([0]))
        # df = df.with_columns('Capacity (Ah)', (dQ_charge - dQ_discharge).cumsum())
        # df = df.with_columns('Capacity (Ah)', df['Capacity (Ah)'] + df['Charge Capacity (Ah)'].max())
        # df = df.with_columns('Exp Capacity (Ah)', pl.Series([0] * len(df)))
        # df = df.with_columns('Cycle Capacity (Ah)', pl.Series([0] * len(df)))
        # df = df.with_columns('Date', pl.to_datetime(df['Date']))
        # df = df.with_column('Time', pl.to_timedelta(df['Time']).dt.total_seconds())
        # return df

    @staticmethod
    def convert_units(df):
        conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}
        for column in df.columns:
            match = re.search(r'\((.*?)\)', column)  # Update the regular expression pattern to extract the unit from the column name
            if match:
                unit = match.group(1)
                prefix = next((x for x in unit if not x.isupper()), None)
                if prefix in conversion_dict:
                    #print(column)
                    df.with_columns(pl.col(column) * conversion_dict[prefix]) 
                    df.rename({column: column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')})
                    #print(df.collect())
        return df