import numpy as np
def test_cycle(BreakinCycles_fixture):
    cycle = BreakinCycles_fixture.cycle(0)
    assert cycle.cycles_idx == 1
    assert cycle.steps_idx == [4, 5, 6, 7]


def test_pulse(Pulsing_fixture):
    pulse = Pulsing_fixture.pulse(0)
    assert pulse.steps_idx == 10
    assert (pulse.RawData['Cycle'] == 5).all()
    
def test_V0(Pulsing_fixture):
    assert Pulsing_fixture.V0[0] == 4.1919
    
def test_V1(Pulsing_fixture):
    assert Pulsing_fixture.V1[0] == 4.1558
    
def test_I1(Pulsing_fixture):
    assert Pulsing_fixture.I1[0] == -0.0199936
    
def test_R0(Pulsing_fixture):
    assert np.isclose(Pulsing_fixture.R0[0], (4.1558-4.1919)/-0.0199936)
    
def test_Rt(Pulsing_fixture):
    assert np.isclose(Pulsing_fixture.Rt(10)[0], (4.1337-4.1919)/-0.0199936)