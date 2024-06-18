"""Script to create a Streamlit dashboard for PyProBE."""
import pickle
from typing import List

import pandas as pd
import plotly
import streamlit as st
from ordered_set import OrderedSet

from pyprobe.plot import Plot

if __name__ == "__main__":
    with open("dashboard_data.pkl", "rb") as f:
        cell_list = pickle.load(f)

    st.title("PyProBE Dashboard")
    st.sidebar.title("Select data to plot")

    info_list = []
    for i in range(len(cell_list)):
        info_list.append(cell_list[i].info)
    info = pd.DataFrame(info_list)

    def dataframe_with_selections(df: pd.DataFrame) -> List[int]:
        """Create a dataframe with a selection column for user input.

        Args:
            df (pd.DataFrame): The dataframe to display.

        Returns:
            list: The list of selected row indices.
        """
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
        selected_indices = (
            selected_rows.index.tolist()
        )  # Get the indices of the selected rows
        return selected_indices

    # Display the DataFrame in the sidebar
    selected_indices = dataframe_with_selections(info)
    # Get the names of the selected rows
    selected_names = [cell_list[i].info["Name"] for i in selected_indices]
    # Get the procedure names from the selected cells
    procedure_names_sets = [
        OrderedSet(cell_list[i].procedure.keys()) for i in selected_indices
    ]

    # Find the common procedure names
    if len(procedure_names_sets) == 0:
        procedure_names = []
    else:
        procedure_names = procedure_names_sets[0]
        for s in procedure_names_sets[1:]:
            procedure_names = [x for x in procedure_names if x in s]

    selected_raw_data = st.sidebar.selectbox("Select a procedure", procedure_names)

    # Select an experiment
    if selected_raw_data is not None:
        experiment_names = (
            cell_list[selected_indices[0]].procedure[selected_raw_data].titles.keys()
        )
        selected_experiment = st.sidebar.multiselect(
            "Select an experiment", experiment_names
        )
        selected_experiment = tuple(selected_experiment)

    # Get the cycle and step numbers from the user
    cycle_step_input = st.sidebar.text_input(
        'Enter the cycle and step numbers (e.g., "cycle(1).step(2)")'
    )
    x_options = [
        "Time [s]",
        "Time [min]",
        "Time [hr]",
        "Capacity [Ah]",
        "Capacity [mAh]",
        "Capacity Throughput [Ah]",
    ]
    y_options = [
        "Voltage [V]",
        "Current [A]",
        "Current [mA]",
        "Capacity [Ah]",
        "Capacity [mAh]",
    ]

    graph_placeholder = st.empty()

    col1, col2, col3 = st.columns(3)
    # Create select boxes for the x and y axes
    x_axis = col1.selectbox("x axis", x_options, index=0)
    y_axis = col2.selectbox("y axis", y_options, index=1)
    secondary_y_axis = col3.selectbox("Secondary y axis", ["None"] + y_options, index=0)

    # Select plot theme

    themes = list(plotly.io.templates)
    themes.remove("none")
    themes.remove("streamlit")
    themes.insert(0, "default")
    plot_theme = "simple_white"

    # Create a figure
    fig = Plot()

    selected_data = []
    for i in range(len(selected_indices)):
        selected_index = selected_indices[i]
        if len(selected_experiment) == 0:
            experiment_data = cell_list[selected_index].procedure[selected_raw_data]
        else:
            experiment_data = (
                cell_list[selected_index]
                .procedure[selected_raw_data]
                .experiment(*selected_experiment)
            )
        # Check if the input is not empty
        if cycle_step_input:
            # Use eval to evaluate the input as Python code
            filtered_data = eval(f"experiment_data.{cycle_step_input}")
        else:
            filtered_data = experiment_data

        if secondary_y_axis == "None":
            secondary_y_axis = None

        fig = fig.add_line(filtered_data, x_axis, y_axis, secondary_y=secondary_y_axis)
        filtered_data = filtered_data.data.to_pandas()
        selected_data.append(filtered_data)

    # Show the plot
    if len(selected_data) > 0 and len(procedure_names) > 0:
        graph_placeholder.plotly_chart(
            fig.fig, theme="streamlit" if plot_theme == "default" else None
        )

    # Show raw data in tabs
    if selected_data:
        tabs = st.tabs(selected_names)
        columns = [
            "Time [s]",
            "Cycle",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
        for tab in tabs:
            tab.dataframe(selected_data[tabs.index(tab)][columns], hide_index=True)
