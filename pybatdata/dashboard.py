import streamlit as st
import pandas as pd
import plotly.express as px

def dashboard(df):
    # Convert Polars dataframe to Pandas if necessary
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    # Create a Streamlit app
    st.title('My Dashboard')

    # Display the raw data in a table
    st.subheader('Raw data')
    st.write(df)

    # Allow the user to select columns to plot
    st.subheader('Interactive plot')
    x_column_to_plot = st.selectbox('Select column for x-axis', df.columns)
    y_column_to_plot = st.selectbox('Select column for y-axis', df.columns)
    fig = px.scatter(df, x=x_column_to_plot, y=y_column_to_plot, title=f'Scatter plot of {y_column_to_plot} vs {x_column_to_plot}')
    st.plotly_chart(fig)