
def test_cycle(BreakinCycles_fixture):
    cycle = BreakinCycles_fixture.cycle(0)
    assert cycle.cycles_idx == 1
    assert cycle.steps_idx == [4, 5, 6, 7]
    assert cycle.step_names ==  [None, 
                                    'Rest', 
                                    'CCCV Chg', 
                                    'Rest', 
                                    'CC DChg', 
                                    'Rest', 
                                    'CCCV Chg', 
                                    'Rest', 
                                    None, 
                                    'Rest', 
                                    'CC DChg', 
                                    'Rest', 
                                    'Rest']
