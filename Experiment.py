

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
    
class Pulsing(Experiment):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
    
    def pulse(self, pulse_number):
        return self.cycle(pulse_number)
    
    @property
    def R0(self):
        for pulse_num in range(1, len(self.cycles_idx)+1):
            if (self.pulse(pulse_num).RawData['Current (A)']).all() >= 0:
                return self.pulse(pulse_num).charge(1).R0
            elif (self.pulse(pulse_num).RawData['Current (A)']).all() <= 0:
                return self.pulse(pulse_num).discharge(1).R0
    
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

    def charge(self, charge_number):
        charge_steps = [index for index, item in enumerate(self.step_names) if item == 'CC Chg']
        charge_steps = list(set(charge_steps) & set(flatten(self.steps_idx)))
        return Charge(self.data, self.cycles_idx, charge_steps[charge_number-1], self.step_names)
 
    def discharge(self, discharge_number):
        discharge_steps = [index for index, item in enumerate(self.step_names) if item == 'CC DChg']
        discharge_steps = list(set(discharge_steps) & set(flatten(self.steps_idx)))
        return Discharge(self.data, self.cycles_idx, discharge_steps[discharge_number-1], self.step_names)
    
    def rest(self, rest_number):
        rest_steps = [index for index, item in enumerate(self.step_names) if item == 'Rest']
        rest_steps = list(set(rest_steps) & set(flatten(self.steps_idx)))
        return Rest(self.data, self.cycles_idx, rest_steps[rest_number-1], self.step_names)
    
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
    
    @property
    def R0(self):
        V1 = self.RawData['Voltage (V)'].iloc[0]
        V2 = self.RawData['Voltage (V)'].loc[self.RawData['Voltage (V)'] != self.RawData['Voltage (V)'].iloc[0]].iloc[0]
        I = self.RawData['Current (A)'].iloc[0]
        return (V2-V1)/I
    
class Charge(Step):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
    
class Discharge(Step):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
        
class Rest(Step):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
    
def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])