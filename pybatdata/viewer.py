import matplotlib.pyplot as plt
class Viewer:
    def __init__(self, data, info):
        self.data = data
        self.info = info

    def plot(self, x, y):
        plt.plot(self.data[x], self.data[y], color = self.info['color'], label=self.info['Name'])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()