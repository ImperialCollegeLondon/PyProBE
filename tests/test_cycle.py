def test_step(BreakinCycles_fixture):
    step = BreakinCycles_fixture.cycle(0).step(1)
    assert (step.data['Step']==5).all()
    
def test_charge(BreakinCycles_fixture):
    charge = BreakinCycles_fixture.cycle(0).charge(0)
    assert (charge.data['Step']==6).all()
    assert (charge.data['Current (A)']>0).all() 
    
def test_discharge(BreakinCycles_fixture):
    discharge = BreakinCycles_fixture.cycle(0).discharge(0)
    assert (discharge.data['Step']==4).all()
    assert (discharge.data['Current (A)']<0).all()
    
def test_chargeordischarge(BreakinCycles_fixture):
    charge = BreakinCycles_fixture.cycle(0).chargeordischarge(0)
    assert (charge.data['Step']==4).all()
    assert (charge.data['Current (A)']<0).all()
    
    discharge = BreakinCycles_fixture.cycle(0).chargeordischarge(1)
    assert (discharge.data['Step']==6).all()
    assert (discharge.data['Current (A)']>0).all()
    
def test_rest(BreakinCycles_fixture):
    rest = BreakinCycles_fixture.cycle(0).rest(0)
    assert (rest.data['Step']==5).all()
    assert (rest.data['Current (A)']==0).all()
    
    rest = BreakinCycles_fixture.cycle(0).rest(1)
    assert (rest.data['Step']==7).all()
    assert (rest.data['Current (A)']==0).all()
    