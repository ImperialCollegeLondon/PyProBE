import matplotlib.pyplot as plt
import polars as pl
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale
import numpy as np
class Viewer:
    def __init__(self, data, info):
        self._data = data
        self.info = info

    @property
    def data(self):
        if isinstance(self._data, pl.LazyFrame):
            return self._data.collect()
        else :
            return self._data

    def plot(self, x, y):
        plt.plot(self.data[x], self.data[y], color = self.info['color'], label=self.info['Name'])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
    
    def print(self):
        print(self.data)

    def plotly(self, 
               fig, 
               x, 
               y, 
               color_by=None, 
               legend_by='Name', 
               colormap='viridis'):
        if color_by is not None:
            if color_by in self.data.columns:
                unique_colors = self.data[color_by].unique().to_numpy()
                colors = sample_colorscale(colormap, minmax_scale(unique_colors))
                for i, condition in enumerate(unique_colors):
                    subset = self.data.filter(pl.col(color_by) == condition)
                    fig.add_trace(go.Scatter(x=subset[x], 
                                             y=subset[y], 
                                             mode='lines', 
                                             line=dict(color=colors[i]), 
                                             name=str(unique_colors[i]),
                                             showlegend=False
                                             ))
                    
                # Dummy heatmap for colorbar
                fig.add_trace(
                    go.Heatmap(
                        z=[[unique_colors.min(), unique_colors.max()]],  # Set z to the range of the color scale
                        colorscale=colormap,  # choose a colorscale
                        colorbar=dict(title=color_by),  # add colorbar
                        opacity=0,
                    )
                )
            else:
                color = self.info['color']
                fig.add_trace(go.Scatter(x=subset[x], y=subset[y], mode='lines', line=dict(color=color), name=self.info[legend_by]))
                fig.update_layout(showlegend=True,
                                  legend = dict(font = dict(size=axis_font_size)))
        title_font_size = 18
        axis_font_size = 14
        plot_theme = 'simple_white'
        x_range = [self.data[x].min(), self.data[x].max()]
        y_range = [self.data[y].min(), self.data[y].max()]
        x_buffer = 0.05 * (x_range[1] - x_range[0])
        y_buffer = 0.05 * (y_range[1] - y_range[0])
        fig.update_xaxes(range=[x_range[0] - x_buffer, x_range[1] + x_buffer])
        fig.update_yaxes(range=[y_range[0] - y_buffer, y_range[1] + y_buffer])
        fig.update_layout(xaxis_title=x, 
                  yaxis_title=y,
                  template=plot_theme,
                  title_font=dict(size=title_font_size),
                  xaxis_title_font=dict(size=title_font_size),
                  yaxis_title_font=dict(size=title_font_size),
                  xaxis_tickfont=dict(size=axis_font_size),
                  yaxis_tickfont=dict(size=axis_font_size),

                    )
        return fig