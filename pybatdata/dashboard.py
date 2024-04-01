import streamlit as st
import os
import pandas as pd
import pickle
import sys
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import plotly.graph_objects as go

import pandas as pd
# Add the parent directory of pybatdata to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('procedure_dict.pkl', 'rb') as f:
    procedure_dict = pickle.load(f)

metadata = pd.DataFrame(procedure_dict).drop('Data', axis=1)
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
selected_indices = dataframe_with_selections(metadata)

# Get the cycle and step numbers from the user
cycle_step_input = st.text_input('Enter the cycle and step numbers (e.g., "cycle(1).step(2)")')
x_options = ['Time', 'Capacity (Ah)']
y_options = ['Voltage (V)', 'Current (A)', 'Capacity (Ah)']

# Create select boxes for the x and y axes
x_axis = st.selectbox('Select x axis', x_options, index=0)
y_axis = st.selectbox('Select y axis', y_options, index=1)

# Create a figure
p = figure(title='Bokeh Plot', x_axis_label=x_axis, y_axis_label=y_axis)
experiment_names = procedure_dict[0]['Data'].titles.keys()
selected_experiment = st.selectbox('Select an experiment', experiment_names)

selected_names = [procedure_dict[i]['Name'] for i in selected_indices]
# Create a figure
fig = go.Figure()
for selected_index in selected_indices:
    data = procedure_dict[selected_index]['Data'].experiment(selected_experiment)
    # Check if the input is not empty
    if cycle_step_input:
        # Use eval to evaluate the input as Python code
        try:
            data_to_plot = eval(f'data.{cycle_step_input}')
        except Exception as e:
            st.write(f'Error: {e}')
    else: 
        data_to_plot = data

    df = data_to_plot.RawData.to_pandas()


    # Add a line to the plot for each selected index
    fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='lines', name=f'Line {selected_index}'))

# Set the plot's title and labels
fig.update_layout(title='Plotly Plot', xaxis_title=x_axis, yaxis_title=y_axis, template='simple_white')

# Show the plot
st.plotly_chart(fig)    

tabs = st.tabs(selected_names)

