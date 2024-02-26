

class Experiment:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.data = data
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
    
    @property
    def RawData(self):
        return self.data[(self.data['Cycle'].isin(flatten(self.cycles_idx))) & (self.data['Step'].isin(flatten(self.steps_idx)))]

    def cycle(self,cycle_number):
        return Cycle(self.data, self.cycles_idx[cycle_number-1], self.steps_idx[cycle_number-1], self.step_names)
    
class Cycle:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.data = data
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names

    @property
    def RawData(self):
        return self.data[(self.data['Cycle'].isin(flatten(self.cycles_idx))) & (self.data['Step'].isin(flatten(self.steps_idx)))]

    def step(self, step_number):
        return Step(self.data, self.cycles_idx, self.steps_idx[step_number-1], self.step_names)
    
class Step:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.data = data
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names

    @property
    def RawData(self):
        return self.data[(self.data['Cycle'].isin(flatten(self.cycles_idx))) & (self.data['Step'].isin(flatten(self.steps_idx)))]

    @property
    def capacity(self):
        return self.RawData['Capacity (Ah)'].max()
    
def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])