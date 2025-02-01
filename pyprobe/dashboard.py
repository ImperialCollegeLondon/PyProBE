"""Script to create a Streamlit dashboard for PyProBE."""

import copy
import os
import pickle
import platform
import subprocess
from typing import Any, List

import distinctipy
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from pyprobe.cell import Cell


def launch_dashboard(cell_list: List[Cell]) -> None:
    """Function to launch the dashboard for the preprocessed data.

    Args:
        cell_list (list): The list of cell objects to display in the dashboard.
    """
    with open("dashboard_data.pkl", "wb") as f:
        pickle.dump(cell_list, f)

    if platform.system() == "Windows":
        subprocess.Popen(
            [
                "cmd",
                "/c",
                "start",
                "/B",
                "streamlit",
                "run",
                os.path.join(os.path.dirname(__file__), "dashboard.py"),
                ">",
                "nul",
                "2>&1",
            ],
            shell=True,
        )
    elif platform.system() == "Darwin":
        subprocess.Popen(
            [
                "nohup",
                "streamlit",
                "run",
                os.path.join(os.path.dirname(__file__), "dashboard.py"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def _dataframe_with_selections(df: pl.DataFrame) -> List[int]:
    """Create a dataframe with a selection column for user input.

    Args:
        df (pd.DataFrame): The dataframe to display.

    Returns:
        list: The list of selected row indices.
    """
    df = df.to_pandas()
    df_with_selections = copy.deepcopy(df)
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.sidebar.data_editor(
        df_with_selections,
        hide_index=True,  # Keep the index visible
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns.tolist(),
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    selected_indices = (
        selected_rows.index.tolist()
    )  # Get the indices of the selected rows
    return selected_indices


def _get_procedure_names(
    cell_list: List[Cell], selected_indices: List[int]
) -> List[str]:
    """Get the common procedure names from the selected cells."""
    procedure_names_sets = [
        list(cell_list[i].procedure.keys()) for i in selected_indices
    ]

    # Find the common procedure names
    if len(procedure_names_sets) == 0:
        procedure_names: List[str] = []
    else:
        procedure_names = list(procedure_names_sets[0])
        for s in procedure_names_sets[1:]:
            procedure_names = [x for x in procedure_names if x in s]
    return list(procedure_names)


def _get_data_filters(
    cell_list: List[Cell], selected_indices: List[int], procedure_names: List[str]
) -> tuple[Any, tuple[Any, ...], Any]:
    import streamlit as st

    selected_procedure = st.sidebar.selectbox("Select a procedure", procedure_names)

    # Select an experiment
    if selected_procedure is not None:
        experiment_names = (
            cell_list[selected_indices[0]]
            .procedure[selected_procedure]
            .experiment_names
        )
        selected_experiment = st.sidebar.multiselect(
            "Select an experiment", experiment_names
        )
        selected_experiment_tuple = tuple(selected_experiment)
    else:
        selected_experiment_tuple = ()

    # Get the cycle and step numbers from the user
    cycle_step_input = st.sidebar.text_input(
        'Enter the cycle and step numbers (e.g., "cycle(1).step(2)")'
    )
    return selected_procedure, selected_experiment_tuple, cycle_step_input


def _get_graph_inputs(
    info: pl.DataFrame, cell_list: List[Cell], selected_indices: List[int]
) -> tuple[Any, Any, Any, Any, list[Any | None]]:
    """Get the user inputs for the graph."""
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
    col1, col2, col3, col4, col5 = st.columns(5)
    # Create select boxes for the x and y axes
    filter_stage = col1.selectbox(
        "Filter stage", ["", "Experiment", "Cycle", "Step"], index=0
    )
    x_axis = col2.selectbox("x axis", x_options, index=0)
    x_axis = f"{filter_stage} {x_axis}".strip()
    y_axis = col3.selectbox("y axis", y_options, index=1)
    secondary_y_axis = col4.selectbox("Secondary y axis", ["None"] + y_options, index=0)
    # choose a cell identifier
    cell_identifier = col5.selectbox("Legend label", info.collect_schema().names())
    selected_names = [cell_list[i].info[cell_identifier] for i in selected_indices]
    return x_axis, y_axis, secondary_y_axis, cell_identifier, selected_names


def _main() -> None:
    """Main function to run the Streamlit dashboard."""
    with open("dashboard_data.pkl", "rb") as f:
        cell_list = pickle.load(f)

    st.title("PyProBE Dashboard")
    st.sidebar.title("Select data to plot")

    info_list = []
    for i in range(len(cell_list)):
        info_list.append(cell_list[i].info)
    info = pl.DataFrame(info_list)

    # Display the DataFrame in the sidebar
    selected_indices = _dataframe_with_selections(info)
    procedure_names = _get_procedure_names(cell_list, selected_indices)

    selected_procedure, selected_experiment_tuple, cycle_step_input = _get_data_filters(
        cell_list, selected_indices, procedure_names
    )
    x_axis, y_axis, secondary_y_axis, cell_identifier, selected_names = (
        _get_graph_inputs(info, cell_list, selected_indices)
    )

    graph_placeholder = st.empty()

    # Create a figure
    fig = go.Figure()
    colors = distinctipy.get_colors(len(cell_list), rng=0)
    selected_data = []
    for i in range(len(selected_indices)):
        selected_index = selected_indices[i]
        if len(selected_experiment_tuple) == 0:
            experiment_data = cell_list[selected_index].procedure[selected_procedure]
        else:
            experiment_data = (
                cell_list[selected_index]
                .procedure[selected_procedure]
                .experiment(*selected_experiment_tuple)
            )
        # Check if the input is not empty
        if cycle_step_input:
            # Use eval to evaluate the input as Python code
            filtered_data = eval(f"experiment_data.{cycle_step_input}")
        else:
            filtered_data = experiment_data

        if secondary_y_axis == "None":
            secondary_y_axis = None
        filtered_data = filtered_data.data.to_pandas()

        # Add primary y-axis data
        primary_trace = go.Scatter(
            x=filtered_data[x_axis],
            y=filtered_data[y_axis],
            mode="lines",
            name=f"{cell_list[selected_index].info[cell_identifier]}",
            line=dict(color=distinctipy.get_hex(colors[selected_index])),
        )
        fig.add_trace(primary_trace)

        # Add secondary y-axis data if specified
        if secondary_y_axis:
            secondary_trace = go.Scatter(
                x=filtered_data[x_axis],
                y=filtered_data[secondary_y_axis],
                mode="lines",
                name=f"{cell_list[selected_index].info[cell_identifier]}",
                yaxis="y2",
                line=dict(
                    color=distinctipy.get_hex(colors[selected_index]), dash="dash"
                ),  # Use the same color as the primary trace
                showlegend=False,
            )
            fig.add_trace(secondary_trace)

        title_font_size = 18
        axis_font_size = 14
        default_layout = go.Layout(
            template="simple_white",
            title_font=dict(size=title_font_size),
            xaxis_title_font=dict(size=title_font_size),
            yaxis_title_font=dict(size=title_font_size),
            xaxis_tickfont=dict(size=axis_font_size),
            yaxis_tickfont=dict(size=axis_font_size),
            legend_font=dict(size=axis_font_size),
            legend=dict(x=1.2),
            width=800,
            height=600,
        )

        # Update layout for dual-axis
        fig.update_layout(
            yaxis=dict(
                title=y_axis,
            ),
            yaxis2=dict(title=secondary_y_axis, overlaying="y", side="right"),
            xaxis=dict(
                title=x_axis,
            ),
        )
        fig.update_layout(default_layout)

        selected_data.append(filtered_data)

    if secondary_y_axis:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name=secondary_y_axis,
                showlegend=True,
            )
        )

    # Show the plot
    if len(selected_data) > 0 and len(procedure_names) > 0:
        graph_placeholder.plotly_chart(
            fig,
            theme="streamlit",  # if plot_theme == "default" else None
        )

    # Show raw data in tabs
    if selected_data:
        tabs = st.tabs(selected_names)
        columns = [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
        for tab in tabs:
            tab.dataframe(selected_data[tabs.index(tab)][columns], hide_index=True)


if __name__ == "__main__":
    _main()
