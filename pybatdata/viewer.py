import matplotlib.pyplot as plt
import polars as pl
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale
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

    def plotly(self, fig, x, y, color_by=None, legend_by='Name', colormap='viridis'):
        if color_by is not None:
            unique_colors = self.data[color_by].unique().to_numpy()
            colors = sample_colorscale(colormap, minmax_scale(unique_colors))
            for i, condition in enumerate(unique_colors):
                subset = self.data.filter(pl.col(color_by) == condition)
                fig.add_trace(go.Scatter(x=subset[x], y=subset[y], mode='lines', line=dict(color=colors[i]), name=self.info[legend_by]))
        return fig