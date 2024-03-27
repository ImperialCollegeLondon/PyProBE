import numpy as np

def test_capacity(BreakinCycles_fixture):
    capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    assert np.isclose(capacity, BreakinCycles_fixture.cycle(0).charge(0).RawData['Charge Capacity (Ah)'].max())
