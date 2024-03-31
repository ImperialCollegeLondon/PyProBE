#%%

import streamlit as st
import os
import pandas as pd
import pickle
import sys
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show
import pandas as pd
# Add the parent directory of pybatdata to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('procedure_dict.pkl', 'rb') as f:
    procedure_dict = pickle.load(f)

metadata = pd.DataFrame(procedure_dict).drop('Data', axis=1)
# Display the DataFrame in the sidebar
st.sidebar.write(metadata)

# Create a selectbox for each row in the DataFrame
options = [f'Row {i}' for i in range(len(metadata))]
selected_row = st.selectbox('Select a row', options)

# Convert the selected row label back to an index
selected_index = int(selected_row.split(' ')[1])

experiment_names = procedure_dict[selected_index]['Data'].titles.keys()
selected_experiment = st.selectbox('Select an experiment', experiment_names)
data = procedure_dict[selected_index]['Data'].experiment(selected_experiment)


# Get the cycle and step numbers from the user
cycle_step_input = st.text_input('Enter the cycle and step numbers (e.g., "cycle(1).step(2)")')

# Check if the input is not empty
if cycle_step_input:
    # Use eval to evaluate the input as Python code
    try:
        data_to_plot = eval(f'data.{cycle_step_input}')
    except Exception as e:
        st.write(f'Error: {e}')
else: 
    data_to_plot = data
    


# Get the column names
columns = data_to_plot.RawData.to_pandas().columns.tolist()

x_options = ['Time', 'Capacity (Ah)']
y_options = ['Voltage (V)', 'Current (A)', 'Capacity (Ah)']

# Create select boxes for the x and y axes
x_axis = st.selectbox('Select x axis', x_options, index=0)
y_axis = st.selectbox('Select y axis', y_options, index=1)

# Create a line chart with the selected axes
# Convert the data to a pandas DataFrame
df = data_to_plot.RawData.to_pandas()

# Create a ColumnDataSource from the DataFrame
source = ColumnDataSource(df)

# Create a figure
p = figure(title='Bokeh Plot', x_axis_label=x_axis, y_axis_label=y_axis)

# Add a line plot to the figure
line = p.line(x=x_axis, y=y_axis, source=source)
# Create a hover tool
# Create a hover tool
hover = HoverTool(tooltips=[
    ("Time", "@Time"),
    ("Voltage (V)", "@{Voltage (V)}"),
    ("Current (A)", "@{Current (A)}"),
    ("Capacity (Ah)", "@{Capacity (Ah)}")
], renderers=[line])

# Add the hover tool to the figure
p.add_tools(hover)
# Show the plot
st.bokeh_chart(p)


st.write(data_to_plot.RawData.to_pandas())
    #%%

