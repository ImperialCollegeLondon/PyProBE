#%%
import pandas as pd
import os
import sys
import time
from import_data import Neware
import json

def save_to_cell(folder_path, test_name):
    # import the experiment record file
    record_xlsx = os.path.join(folder_path, "Experiment_Record.xlsx")
    record = pd.read_excel(record_xlsx, sheet_name = test_name)
    
    for i in range(len(record)): # for each line in record
        t1 = time.time()
        # load the metadata
        test_details = record.iloc[i].to_dict()
        cell_num = int(test_details['Cell'])
        cycler = int(test_details['Cycler']) if not pd.isna(test_details['Cycler']) else None
        channel = int(test_details['Channel']) if not pd.isna(test_details['Channel']) else None
        data_path = os.path.join(folder_path, f"Cell_{cell_num}", f'{test_name}.parquet')
        metadata_path = os.path.join(folder_path, f"Cell_{cell_num}", f'{test_name}_details.json')
        test_data = Neware.import_csv(os.path.join(folder_path, test_name, f"{test_name}_{cycler}_{channel}.csv"))
        # create directory for cell if it doesn't exist
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        # save data to parquet
        test_data.to_parquet(data_path, engine='pyarrow')
        # save metadata to json
        with open(metadata_path, 'w') as f:
            json.dump(test_details, f)
        print(f"Processed: Cell {cell_num} in {time.time()-t1:.2f} seconds. File size = {os.path.getsize(data_path)/1e6:.2f} MB") 

    
#folder_path = input("Please enter the folder path: ")
folder_path = "/mnt/c/Users/tjh17/OneDrive - Imperial College London/PhD Workspace/05 - Parallel SLP/DATA"
#experiment = input("Please enter the experiment name:")
test_name = "SLP_Cell-Cell_Variation_R1"

save_to_cell(folder_path, test_name)
# %%
import cProfile
cProfile.run('save_to_cell(folder_path, test_name)', sort='cumtime')
# %%
