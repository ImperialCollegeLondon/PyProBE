"""Tests for the filter module."""

import numpy as np
import polars as pl
import pytest

import pyprobe.filters as filters


def test_step(BreakinCycles_fixture, benchmark):
    """Test the step method."""

    def step():
        return BreakinCycles_fixture.cycle(0).step(1).data

    data = benchmark(step)
    assert (data["Step"] == 5).all()


def test_multi_step(BreakinCycles_fixture, benchmark):
    """Test the step method."""

    def multi_step():
        return BreakinCycles_fixture.cycle(0).step(range(1, 4)).data

    data = benchmark(multi_step)
    assert (data["Step"].unique() == [5, 6, 7]).all()


def test_charge(BreakinCycles_fixture, benchmark):
    """Test the charge method."""

    def charge():
        return BreakinCycles_fixture.cycle(0).charge(0).data

    data = benchmark(charge)
    assert (data["Step"] == 6).all()
    assert (data["Current [A]"] > 0).all()


def test_discharge(BreakinCycles_fixture, benchmark):
    """Test the discharge method."""

    def discharge():
        return BreakinCycles_fixture.cycle(0).discharge(0).data

    data = benchmark(discharge)
    assert (data["Step"] == 4).all()
    assert (data["Current [A]"] < 0).all()

    # test invalid input
    with pytest.raises(ValueError):
        BreakinCycles_fixture.cycle(6).data


def test_chargeordischarge(BreakinCycles_fixture, benchmark):
    """Test the chargeordischarge method."""

    def chargeordischarge():
        return BreakinCycles_fixture.cycle(0).chargeordischarge(0).data

    data = benchmark(chargeordischarge)
    assert (data["Step"] == 4).all()
    assert (data["Current [A]"] < 0).all()

    data = BreakinCycles_fixture.cycle(0).chargeordischarge(1).data
    assert (data["Step"] == 6).all()
    assert (data["Current [A]"] > 0).all()


def test_rest(BreakinCycles_fixture, benchmark):
    """Test the rest method."""

    def rest():
        return BreakinCycles_fixture.cycle(0).rest(0).data

    data = benchmark(rest)
    assert (data["Step"] == 5).all()
    assert (data["Current [A]"] == 0).all()

    data = BreakinCycles_fixture.cycle(0).rest(1).data
    assert (data["Step"] == 7).all()
    assert (data["Current [A]"] == 0).all()


def test_negative_cycle_index(BreakinCycles_fixture, benchmark):
    """Test the negative index."""

    def negative_cycle_index():
        return BreakinCycles_fixture.cycle(-1).data

    data = benchmark(negative_cycle_index)
    assert (data["Cycle"] == 4).all()
    assert (data["Step"].unique() == [4, 5, 6, 7]).all()


def test_negative_step_index(BreakinCycles_fixture, benchmark):
    """Test the negative index."""

    def negative_step_index():
        return BreakinCycles_fixture.cycle(0).step(-1).data

    data = benchmark(negative_step_index)
    assert (data["Step"] == 7).all()


def test_cycle(BreakinCycles_fixture, benchmark):
    """Test the cycle method."""

    def cycle():
        return BreakinCycles_fixture.cycle(2).data

    data = benchmark(cycle)
    assert (data["Cycle"] == 2).all()
    assert (data["Step"].unique() == [4, 5, 6, 7]).all()

    assert data["Cycle Time [s]"][0] == 0
    assert data["Cycle Capacity [Ah]"][0] == 0


def test_constant_current(BreakinCycles_fixture, benchmark):
    """Test backward-compatible constant_current with no target/rtol."""

    def constant_current():
        return BreakinCycles_fixture.constant_current(1).data

    data = benchmark(constant_current)
    assert len(data) > 0
    # No-target branch uses global mode of non-zero current; all returned rows
    # must be within 0.1% of that mode.
    mode_val = data["Current [A]"].mode()[0]
    assert ((data["Current [A]"] - mode_val).abs() <= abs(mode_val) * 0.001).all()


def test_constant_voltage(BreakinCycles_fixture, benchmark):
    """Test backward-compatible constant_voltage with no target/rtol."""

    def constant_voltage():
        return BreakinCycles_fixture.constant_voltage(1).data

    data = benchmark(constant_voltage)
    assert len(data) > 0
    # No-target branch uses global mode; all rows must be within 0.1% of mode.
    mode_val = data["Voltage [V]"].mode()[0]
    assert ((data["Voltage [V]"] - mode_val).abs() <= abs(mode_val) * 0.001).all()


def test_all_steps(BreakinCycles_fixture, benchmark):
    """Test the all_steps method."""

    def all_steps():
        return BreakinCycles_fixture.cycle(0).step().data

    data = benchmark(all_steps)
    assert (data["Cycle"] == 0).all()
    assert (data["Step"].unique() == [4, 5, 6, 7]).all()


def test_zeroed_columns(BreakinCycles_fixture):
    """Test the zeroed_columns method."""
    exp_filtered_data = BreakinCycles_fixture
    cycle_filtered_data = BreakinCycles_fixture.cycle(0)
    step_filtered_data = BreakinCycles_fixture.cycle(0).step(0)

    assert exp_filtered_data.get("Experiment Time [s]")[0] == 0
    assert exp_filtered_data.get("Experiment Capacity [Ah]")[0] == 0
    assert cycle_filtered_data.get("Cycle Time [s]")[0] == 0
    assert cycle_filtered_data.get("Cycle Capacity [Ah]")[0] == 0
    assert step_filtered_data.get("Step Time [s]")[0] == 0
    assert step_filtered_data.get("Step Capacity [Ah]")[0] == 0


@pytest.fixture
def generic_experiment():
    """Return a generic filter."""
    steps = [
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
    ]
    dataframe = pl.DataFrame(
        {
            "Time [s]": list(range(len(steps))),
            "Step": steps,
            "Event": list(range(len(steps))),
            "Current [A]": steps,
            "Voltage [V]": steps,
            "Capacity [Ah]": steps,
        },
    )
    info = {}
    step_descriptions = {
        "Step": [0, 1, 2, 3],
        "Description": ["Charge", "Discharge", "Charge", "Discharge"],
    }

    cycle_info = [(0, 3, 2), (0, 1, 2)]
    return filters.Experiment(
        lf=dataframe,
        info=info,
        step_descriptions=step_descriptions,
        cycle_info=cycle_info,
    )


def test_cycle_generic(generic_experiment):
    """Test the cycle method."""
    assert generic_experiment.cycle_info == [(0, 3, 2), (0, 1, 2)]
    assert filters._cycle(generic_experiment, 0).data[
        "Time [s]"
    ].unique().to_list() == list(range(26))
    assert filters._cycle(generic_experiment, 1).data[
        "Time [s]"
    ].unique().to_list() == list(range(26, 52))
    assert filters._cycle(generic_experiment, -1).data[
        "Time [s]"
    ].unique().to_list() == list(range(26, 52))

    next_cycle = filters._cycle(generic_experiment, 1)
    assert next_cycle.cycle_info == [(0, 1, 2)]
    assert filters._cycle(next_cycle, 0).data["Time [s]"].unique().to_list() == list(
        range(26, 31),
    )
    assert filters._cycle(next_cycle, 3).data["Time [s]"].unique().to_list() == list(
        range(41, 46),
    )
    assert filters._cycle(next_cycle, -1).data["Time [s]"].unique().to_list() == list(
        range(46, 52),
    )

    # test when cycle numbers are inferred
    generic_experiment.cycle_info = []
    assert filters._cycle(generic_experiment, 0).data[
        "Time [s]"
    ].unique().to_list() == list(range(5))
    assert filters._cycle(generic_experiment, -1).data[
        "Time [s]"
    ].unique().to_list() == list(range(41, 52))


def _make_multilevel_experiment(
    col: str,
    level_a: float,
    level_b: float,
    rows_a: int = 100,
    rows_b: int = 50,
) -> filters.Experiment:
    """Return an Experiment with alternating constant-value holds.

    Structure: level_a (rows_a rows), level_b (rows_b rows),
               level_a (rows_a rows), level_b (rows_b rows).
    level_a has more rows so it is the global mode when no target is given.
    """
    levels = [level_a, level_b, level_a, level_b]
    counts = [rows_a, rows_b, rows_a, rows_b]
    values = []
    events = []
    for ev, (val, n) in enumerate(zip(levels, counts)):
        values.extend([val] * n)
        events.extend([ev] * n)

    n_total = len(values)
    other = [0.0] * n_total
    df = pl.DataFrame(
        {
            "Time [s]": list(range(n_total)),
            "Step": [0] * n_total,
            "Event": events,
            "Current [A]": values if col == "Current [A]" else other,
            "Voltage [V]": values if col == "Voltage [V]" else [3.7] * n_total,
            "Capacity [Ah]": other,
        }
    )
    return filters.Experiment(
        lf=df,
        info={},
        step_descriptions={"Step": [0], "Description": ["Test"]},
        cycle_info=[],
    )


@pytest.fixture
def multilevel_cv():
    """Experiment with CV holds at 4.2 V (dominant, 100 rows) and 3.6 V (50 rows)."""
    return _make_multilevel_experiment("Voltage [V]", 4.2, 3.6)


@pytest.fixture
def multilevel_cc():
    """Experiment with CC steps at 1.0 A (dominant, 100 rows) and 2.0 A (50 rows)."""
    return _make_multilevel_experiment("Current [A]", 1.0, 2.0)


def test_constant_voltage_with_target(multilevel_cv):
    """constant_voltage(target=X) returns only rows within target ± |target|*rtol."""
    data = multilevel_cv.constant_voltage(0, target=4.2, rtol=0.001).data
    assert data["Voltage [V]"].min() >= 4.2 * 0.999
    assert data["Voltage [V]"].max() <= 4.2 * 1.001
    assert len(data) == 100
    # Negative target does not match positive voltages
    assert multilevel_cv.constant_voltage(target=-4.2).lf.collect().is_empty()


def test_constant_current_with_target(multilevel_cc):
    """constant_current(target=X) returns only rows within target ± |target|*rtol."""
    data = multilevel_cc.constant_current(0, target=1.0, rtol=0.001).data
    assert data["Current [A]"].min() >= 1.0 * 0.999
    assert data["Current [A]"].max() <= 1.0 * 1.001
    assert len(data) == 100
    # Negative target (discharge) does not match positive values (charge)
    assert multilevel_cc.constant_current(target=-1.0).lf.collect().is_empty()


def test_constant_voltage_no_target_dominant_level(multilevel_cv):
    """constant_voltage() without target returns only the globally dominant level."""
    data_0 = multilevel_cv.constant_voltage(0).data
    data_1 = multilevel_cv.constant_voltage(1).data
    assert np.isclose(data_0["Voltage [V]"].mean(), 4.2, rtol=0.001)
    assert np.isclose(data_1["Voltage [V]"].mean(), 4.2, rtol=0.001)
    with pytest.raises(ValueError):
        multilevel_cv.constant_voltage(2).data


def test_constant_voltage_tolerance_controls_band():
    """Rtol controls band width when target is given."""
    df = pl.DataFrame(
        {
            "Time [s]": list(range(20)),
            "Step": [0] * 20,
            "Event": [0] * 10 + [1] * 10,
            "Current [A]": [0.0] * 20,
            "Voltage [V]": [4.2] * 10 + [4.19] * 10,
            "Capacity [Ah]": [0.0] * 20,
        }
    )
    exp = filters.Experiment(
        lf=df,
        info={},
        step_descriptions={"Step": [0], "Description": ["Test"]},
        cycle_info=[],
    )
    # rtol=0.003: band [4.1874, 4.2126] — includes both 4.2 V and 4.19 V
    wide = exp.constant_voltage(target=4.2, rtol=0.003).data
    # rtol=0.001: band [4.1958, 4.2042] — excludes 4.19 V
    narrow = exp.constant_voltage(target=4.2, rtol=0.001).data
    assert len(wide) == 20
    assert len(narrow) == 10
    assert narrow["Voltage [V]"].min() == 4.2


def test_constant_voltage_index_with_target(multilevel_cv):
    """constant_voltage(0, target=X) returns only the first matching hold."""
    first = multilevel_cv.constant_voltage(0, target=4.2, rtol=0.001).data
    both = multilevel_cv.constant_voltage(target=4.2, rtol=0.001).data
    assert len(first) < len(both)
    assert len(first) == 100
    assert first["Voltage [V]"].min() >= 4.2 * 0.999
