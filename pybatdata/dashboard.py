import streamlit as st
import os
import pandas as pd
import pickle
import sys
import plotly.graph_objects as go
import plotly

# Add the parent directory of pybatdata to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pybatdata.plot import Plot

with open('dashboard_data.pkl', 'rb') as f:
    cell_list = pickle.load(f)

st.title('PyBatData Dashboard')
st.sidebar.title('Select data to plot')

info_list = []
for i in range(len(cell_list)):
    info_list.append(cell_list[i].info)
info = pd.DataFrame(info_list)
def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.sidebar.data_editor(
        df_with_selections,
        hide_index=True,  # Keep the index visible
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    selected_indices = selected_rows.index.tolist()  # Get the indices of the selected rows
    return selected_indices

# Display the DataFrame in the sidebar
selected_indices = dataframe_with_selections(info)
# Get the names of the selected rows
selected_names = [cell_list[i].info['Name'] for i in selected_indices]

# Select a procedure
procedure_names = cell_list[0].procedure.keys()
selected_raw_data = st.sidebar.selectbox('Select a procedure', procedure_names)

# Select an experiment
experiment_names = cell_list[0].procedure[selected_raw_data].titles.keys()
selected_experiment = st.sidebar.selectbox('Select an experiment', experiment_names)

# Get the cycle and step numbers from the user
cycle_step_input = st.sidebar.text_input('Enter the cycle and step numbers (e.g., "cycle(1).step(2)")')
x_options = ['Time [s]', 'Time [min]', 'Time [hr]', 'Capacity [Ah]', 'Capacity [mAh]', 'Capacity Throughput [Ah]']
y_options = ['Voltage [V]', 'Current [A]', 'Current [mA]', 'Capacity [Ah]', 'Capacity [mAh]']

graph_placeholder = st.empty()

col1, col2, col3 = st.columns(3)
# Create select boxes for the x and y axes
x_axis = col1.selectbox('x axis', x_options, index=0)
y_axis = col2.selectbox('y axis', y_options, index=1)
secondary_y_axis = col3.selectbox('Secondary y axis', ['None'] + y_options, index=0)

# Select plot theme

themes = list(plotly.io.templates)
themes.remove('none')
themes.remove('streamlit')
themes.insert(0, 'default')
plot_theme = 'simple_white'

# Create a figure
fig = Plot()

selected_data = []
for i in range(len(selected_indices)):
    selected_index = selected_indices[i]
    experiment_data = cell_list[selected_index].procedure[selected_raw_data].experiment(selected_experiment)
    # Check if the input is not empty
    if cycle_step_input:
        # Use eval to evaluate the input as Python code
        filtered_data = eval(f'experiment_data.{cycle_step_input}')
    else: 
        filtered_data = experiment_data
    
    if secondary_y_axis == 'None':
        secondary_y_axis = None

    fig = fig.add_line(filtered_data, x_axis, y_axis, secondary_y=secondary_y_axis)
    filtered_data = filtered_data.data.to_pandas()
    selected_data.append(filtered_data)

# Show the plot
if len(selected_data) > 0:  
    graph_placeholder.plotly_chart(fig.fig, theme='streamlit' if plot_theme == 'default' else None) 

# Show raw data in tabs
if selected_data:
    tabs = st.tabs(selected_names)
    columns = ['Time [s]', 'Cycle', 'Step', 'Current [A]', 'Voltage [V]', 'Capacity [Ah]']
    for tab in tabs:
        tab.dataframe(selected_data[tabs.index(tab)][columns], hide_index=True)
