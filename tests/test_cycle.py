def test_step(BreakinCycles_fixture):
    step = BreakinCycles_fixture.cycle(0).step(1)
    assert (step.raw_data['Step']==5).all()
    
def test_charge(BreakinCycles_fixture):
    charge = BreakinCycles_fixture.cycle(0).charge(0)
    assert (charge.raw_data['Step']==6).all()
    assert (charge.raw_data['Current (A)']>0).all() 
    
def test_discharge(BreakinCycles_fixture):
    discharge = BreakinCycles_fixture.cycle(0).discharge(0)
    assert (discharge.raw_data['Step']==4).all()
    assert (discharge.raw_data['Current (A)']<0).all()
    
def test_chargeordischarge(BreakinCycles_fixture):
    charge = BreakinCycles_fixture.cycle(0).chargeordischarge(0)
    assert (charge.raw_data['Step']==4).all()
    assert (charge.raw_data['Current (A)']<0).all()
    
    discharge = BreakinCycles_fixture.cycle(0).chargeordischarge(1)
    assert (discharge.raw_data['Step']==6).all()
    assert (discharge.raw_data['Current (A)']>0).all()
    
def test_rest(BreakinCycles_fixture):
    rest = BreakinCycles_fixture.cycle(0).rest(0)
    assert (rest.raw_data['Step']==5).all()
    assert (rest.raw_data['Current (A)']==0).all()
    
    rest = BreakinCycles_fixture.cycle(0).rest(1)
    assert (rest.raw_data['Step']==7).all()
    assert (rest.raw_data['Current (A)']==0).all()
    