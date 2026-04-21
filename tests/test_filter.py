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
    """Test the constant current method."""

    def constant_current():
        return BreakinCycles_fixture.constant_current(1, target=0.004).data

    data = benchmark(constant_current)
    assert np.isclose(data["Current [A]"].to_numpy().mean(), 0.004, rtol=0.001)
    assert data["Current [A]"].min() > 0.004 - 0.004 * 0.001
    assert data["Current [A]"].max() < 0.004 + 0.004 * 0.001


def test_constant_voltage(BreakinCycles_fixture, benchmark):
    """Test the constant voltage method."""

    def constant_voltage():
        return BreakinCycles_fixture.constant_voltage(1, target=4.2).data

    data = benchmark(constant_voltage)
    assert np.isclose(data["Voltage [V]"].to_numpy().mean(), 4.2, rtol=0.001)
    assert data["Voltage [V]"].min() > 4.195
    assert data["Voltage [V]"].max() < 4.2


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


def test_slice_positive_only(BreakinCycles_fixture):
    """Test slicing with positive start and stop."""
    data = BreakinCycles_fixture.cycle(0).step(slice(0, 2)).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [4, 5]


def test_slice_negative_only(BreakinCycles_fixture):
    """Test slicing with negative bounds."""
    data = BreakinCycles_fixture.cycle(0).step(slice(-2, None)).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [6, 7]


def test_slice_mixed_bounds(BreakinCycles_fixture):
    """Test slicing with mixed positive and negative bounds."""
    data = BreakinCycles_fixture.cycle(0).step(slice(-3, 3)).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [5, 6]


def test_slice_with_step_greater_than_one(BreakinCycles_fixture):
    """Test slicing with step > 1."""
    data = BreakinCycles_fixture.cycle(0).step(slice(0, None, 2)).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [4, 6]


def test_slice_negative_start_open_ended(BreakinCycles_fixture):
    """Test slice(-n, 0) which should be open-ended from -n to end."""
    data = BreakinCycles_fixture.cycle(0).step(slice(-2, None)).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [6, 7]


def test_slice_empty_result_zero_stop(BreakinCycles_fixture):
    """Test slice(k, 0) for k >= 0 which should give empty result."""
    with pytest.raises(ValueError, match="No data exists for this filter"):
        BreakinCycles_fixture.cycle(0).step(slice(0, 0)).data


def test_slice_empty_result_positive_stop_zero(BreakinCycles_fixture):
    """Test slice(1, 0) which should give empty result."""
    with pytest.raises(ValueError, match="No data exists for this filter"):
        BreakinCycles_fixture.cycle(0).step(slice(1, 0)).data


def test_slice_step_greater_than_one_with_negative_start(BreakinCycles_fixture):
    """Test slice with step > 1 and negative start."""
    data = BreakinCycles_fixture.cycle(0).step(slice(-4, None, 2)).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [4, 6]


def test_include_preceding_point_step(BreakinCycles_fixture):
    """Test include_preceding_point with step filter."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    data_without = cycle0.step(1).data
    data_with = cycle0.step(1, include_preceding_point=True).data

    assert len(data_with) == len(data_without) + 1
    # Check content of the prepended row: it should match the row before
    # Step 1 in Cycle 0
    c0_data = cycle0.data
    # Find the index of the first row of data_without in c0_data
    # We match on Time [s], Step, and Event to be unique
    match = (
        (c0_data["Time [s]"] == data_without["Time [s]"][0])
        & (c0_data["Step"] == data_without["Step"][0])
        & (c0_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == c0_data["Time [s]"][idx - 1]
    assert data_with["Step"][0] == c0_data["Step"][idx - 1]
    # Tail should match
    assert (
        data_with.tail(len(data_without))["Time [s]"].to_list()
        == data_without["Time [s]"].to_list()
    )


def test_include_preceding_point_charge(BreakinCycles_fixture):
    """Test include_preceding_point with charge filter."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    data_without = cycle0.charge(0).data
    data_with = cycle0.charge(0, include_preceding_point=True).data

    assert len(data_with) == len(data_without) + 1
    c0_data = cycle0.data
    match = (
        (c0_data["Time [s]"] == data_without["Time [s]"][0])
        & (c0_data["Step"] == data_without["Step"][0])
        & (c0_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_include_preceding_point_discharge(BreakinCycles_fixture):
    """Test include_preceding_point with discharge filter."""
    # Use discharge(1) at experiment level to ensure a predecessor exists
    # (Step 4 of cycle 1, predecessor is Step 7 of cycle 0)
    data_without = BreakinCycles_fixture.discharge(1).data
    data_with = BreakinCycles_fixture.discharge(1, include_preceding_point=True).data

    assert len(data_with) == len(data_without) + 1
    full_data = BreakinCycles_fixture.data
    match = (
        (full_data["Time [s]"] == data_without["Time [s]"][0])
        & (full_data["Step"] == data_without["Step"][0])
        & (full_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == full_data["Time [s]"][idx - 1]


def test_include_preceding_point_rest(BreakinCycles_fixture):
    """Test include_preceding_point with rest filter."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    # rest(0) is Step 5, predecessor is Step 4
    data_without = cycle0.rest(0).data
    data_with = cycle0.rest(0, include_preceding_point=True).data

    assert len(data_with) == len(data_without) + 1
    c0_data = cycle0.data
    match = (
        (c0_data["Time [s]"] == data_without["Time [s]"][0])
        & (c0_data["Step"] == data_without["Step"][0])
        & (c0_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_include_preceding_point_chargeordischarge(BreakinCycles_fixture):
    """Test include_preceding_point with chargeordischarge filter."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    # chargeordischarge(1) is Step 6, predecessor is Step 5
    data_without = cycle0.chargeordischarge(1).data
    data_with = cycle0.chargeordischarge(1, include_preceding_point=True).data

    assert len(data_with) == len(data_without) + 1
    c0_data = cycle0.data
    match = (
        (c0_data["Time [s]"] == data_without["Time [s]"][0])
        & (c0_data["Step"] == data_without["Step"][0])
        & (c0_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_include_preceding_point_cycle(BreakinCycles_fixture):
    """Test include_preceding_point with cycle filter on cycle 1."""
    data_without = BreakinCycles_fixture.cycle(1).data
    data_with = BreakinCycles_fixture.cycle(1, include_preceding_point=True).data

    assert len(data_with) == len(data_without) + 1
    full_data = BreakinCycles_fixture.data
    match = (
        (full_data["Time [s]"] == data_without["Time [s]"][0])
        & (full_data["Step"] == data_without["Step"][0])
        & (full_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == full_data["Time [s]"][idx - 1]


def test_iter_step_single_index(BreakinCycles_fixture):
    """Test iter_step with a single index."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    results = list(cycle0.iter_step(0))
    assert len(results) == 1
    assert results[0].data["Step"].unique().to_list() == [4]


def test_iter_step_range(BreakinCycles_fixture):
    """Test iter_step with a range of indices."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    results = list(cycle0.iter_step(range(2)))
    assert len(results) == 2
    assert results[0].data["Step"].unique().to_list() == [4]
    assert results[1].data["Step"].unique().to_list() == [5]


def test_iter_step_slice(BreakinCycles_fixture):
    """Test iter_step with a slice."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    results = list(cycle0.iter_step(slice(0, 2)))
    assert len(results) == 2
    assert results[0].data["Step"].unique().to_list() == [4]
    assert results[1].data["Step"].unique().to_list() == [5]


def test_iter_charge(BreakinCycles_fixture):
    """Test iter_charge at experiment level."""
    results = list(BreakinCycles_fixture.iter_charge())
    assert len(results) == 5
    for item in results:
        assert item.data["Step"].unique().to_list() == [6]
        assert (item.data["Current [A]"] > 0).all()


def test_iter_discharge(BreakinCycles_fixture):
    """Test iter_discharge at experiment level."""
    results = list(BreakinCycles_fixture.iter_discharge())
    assert len(results) == 5
    for item in results:
        assert item.data["Step"].unique().to_list() == [4]
        assert (item.data["Current [A]"] < 0).all()


def test_iter_rest(BreakinCycles_fixture):
    """Test iter_rest at experiment level."""
    results = list(BreakinCycles_fixture.iter_rest())
    assert len(results) == 10
    for item in results:
        assert (item.data["Current [A]"] == 0).all()


def test_iter_chargeordischarge(BreakinCycles_fixture):
    """Test iter_chargeordischarge at experiment level."""
    results = list(BreakinCycles_fixture.iter_chargeordischarge())
    assert len(results) == 10
    for item in results:
        assert (item.data["Current [A]"].abs() > 0).all()


def test_iter_cycle(BreakinCycles_fixture):
    """Test iter_cycle."""
    results = list(BreakinCycles_fixture.iter_cycle())
    assert len(results) == 5
    assert [c.data["Cycle"].unique()[0] for c in results] == [0, 1, 2, 3, 4]


def test_iter_step_with_include_preceding_point(BreakinCycles_fixture):
    """Test iter_step with include_preceding_point."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    # Use index 1 in cycle 0 to ensure a predecessor exists
    results_without = list(cycle0.iter_step(slice(1, 2)))
    results_with = list(cycle0.iter_step(slice(1, 2), include_preceding_point=True))

    assert len(results_with) == len(results_without)
    assert len(results_with[0].data) == len(results_without[0].data) + 1
    c0_data = cycle0.data
    first_row = results_without[0].data
    match = (
        (c0_data["Time [s]"] == first_row["Time [s]"][0])
        & (c0_data["Step"] == first_row["Step"][0])
        & (c0_data["Event"] == first_row["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert results_with[0].data["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_iter_charge_with_include_preceding_point(BreakinCycles_fixture):
    """Test iter_charge with include_preceding_point."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    # charge(0) is Step 6, predecessor is Step 5. This should work.
    results_without = list(cycle0.iter_charge(slice(0, 1)))
    results_with = list(cycle0.iter_charge(slice(0, 1), include_preceding_point=True))

    assert len(results_with) == len(results_without)
    assert len(results_with[0].data) == len(results_without[0].data) + 1
    c0_data = cycle0.data
    first_row = results_without[0].data
    match = (
        (c0_data["Time [s]"] == first_row["Time [s]"][0])
        & (c0_data["Step"] == first_row["Step"][0])
        & (c0_data["Event"] == first_row["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert results_with[0].data["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_iter_discharge_with_include_preceding_point(BreakinCycles_fixture):
    """Test iter_discharge with include_preceding_point."""
    # Use index 1 at experiment level to ensure a predecessor exists
    results_without = list(BreakinCycles_fixture.iter_discharge(slice(1, 2)))
    results_with = list(
        BreakinCycles_fixture.iter_discharge(slice(1, 2), include_preceding_point=True)
    )

    assert len(results_with) == len(results_without)
    assert len(results_with[0].data) == len(results_without[0].data) + 1
    full_data = BreakinCycles_fixture.data
    first_row = results_without[0].data
    match = (
        (full_data["Time [s]"] == first_row["Time [s]"][0])
        & (full_data["Step"] == first_row["Step"][0])
        & (full_data["Event"] == first_row["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert results_with[0].data["Time [s]"][0] == full_data["Time [s]"][idx - 1]


def test_iter_rest_with_include_preceding_point(BreakinCycles_fixture):
    """Test iter_rest with include_preceding_point."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    results_without = list(cycle0.iter_rest(slice(0, 1)))
    results_with = list(cycle0.iter_rest(slice(0, 1), include_preceding_point=True))

    assert len(results_with) == len(results_without)
    assert len(results_with[0].data) == len(results_without[0].data) + 1
    c0_data = cycle0.data
    first_row = results_without[0].data
    match = (
        (c0_data["Time [s]"] == first_row["Time [s]"][0])
        & (c0_data["Step"] == first_row["Step"][0])
        & (c0_data["Event"] == first_row["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert results_with[0].data["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_iter_chargeordischarge_with_include_preceding_point(BreakinCycles_fixture):
    """Test iter_chargeordischarge with include_preceding_point."""
    cycle0 = BreakinCycles_fixture.cycle(0)
    # Use index 1 in cycle 0 to ensure a predecessor exists
    results_without = list(cycle0.iter_chargeordischarge(slice(1, 2)))
    results_with = list(
        cycle0.iter_chargeordischarge(slice(1, 2), include_preceding_point=True)
    )

    assert len(results_with) == len(results_without)
    assert len(results_with[0].data) == len(results_without[0].data) + 1
    c0_data = cycle0.data
    first_row = results_without[0].data
    match = (
        (c0_data["Time [s]"] == first_row["Time [s]"][0])
        & (c0_data["Step"] == first_row["Step"][0])
        & (c0_data["Event"] == first_row["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert results_with[0].data["Time [s]"][0] == c0_data["Time [s]"][idx - 1]


def test_iter_cycle_with_include_preceding_point(BreakinCycles_fixture):
    """Test iter_cycle with include_preceding_point on cycle 1."""
    results_without = list(BreakinCycles_fixture.iter_cycle(slice(1, 2)))
    results_with = list(
        BreakinCycles_fixture.iter_cycle(slice(1, 2), include_preceding_point=True)
    )

    assert len(results_with) == len(results_without)
    assert len(results_with[0].data) == len(results_without[0].data) + 1
    full_data = BreakinCycles_fixture.data
    first_row = results_without[0].data
    match = (
        (full_data["Time [s]"] == first_row["Time [s]"][0])
        & (full_data["Step"] == first_row["Step"][0])
        & (full_data["Event"] == first_row["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert results_with[0].data["Time [s]"][0] == full_data["Time [s]"][idx - 1]


@pytest.mark.parametrize("exp_name", ["Break-in Cycles", "Discharge Pulses"])
def test_procedure_experiment_with_include_preceding_point(procedure_fixture, exp_name):
    """Test Procedure.experiment with include_preceding_point."""
    data_without = procedure_fixture.experiment(exp_name).data
    data_with = procedure_fixture.experiment(
        exp_name, include_preceding_point=True
    ).data

    assert len(data_with) == len(data_without) + 1
    full_data = procedure_fixture.data
    match = (
        (full_data["Time [s]"] == data_without["Time [s]"][0])
        & (full_data["Step"] == data_without["Step"][0])
        & (full_data["Event"] == data_without["Event"][0])
    )
    idx = int(np.where(match)[0][0])
    assert idx > 0
    assert data_with["Time [s]"][0] == full_data["Time [s]"][idx - 1]


def test_negative_step_in_slice_raises_error(BreakinCycles_fixture):
    """Test that negative step in slice raises ValueError."""
    with pytest.raises(ValueError, match="Negative step is not supported"):
        BreakinCycles_fixture.cycle(0).step(slice(3, 0, -1)).data


def test_multiple_indices_in_build_mask(BreakinCycles_fixture):
    """Test _build_mask with multiple indices."""
    data = BreakinCycles_fixture.cycle(0).step(0, 1, 2).data
    step_values = sorted(data["Step"].unique().to_list())
    assert step_values == [4, 5, 6]


def test_multiple_indices_with_range_and_slice(BreakinCycles_fixture):
    """Test multiple indices combining range and slice."""
    data = BreakinCycles_fixture.cycle(0).step(0, slice(2, 4)).data
    step_values = set(data["Step"].unique().to_list())
    assert step_values == {4, 6, 7}


@pytest.fixture
def generic_experiment():
    """Return a generic experiment for testing nested cycles."""
    # Outer cycle: Step 1, 2, 1, 2, 3, 4 (26 rows)
    # Inner cycle: Step 1, 2
    # 2 outer cycles total = 52 rows

    # One outer cycle block:
    # Step 1: 3 rows, Step 2: 3 rows
    # Step 1: 3 rows, Step 2: 3 rows
    # Step 3: 10 rows, Step 4: 4 rows
    outer_steps = [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3 + [3] * 10 + [4] * 4
    steps = outer_steps * 2

    events = []
    # 6 steps per outer cycle * 2 outer cycles = 12 events
    event_counts = [3, 3, 3, 3, 10, 4] * 2
    for i, count in enumerate(event_counts):
        events.extend([i] * count)

    currents = []
    for _ in range(2):
        currents.extend([1.0] * 3)  # CC Charge
        currents.extend([0.5, 0.2, 0.1])  # CV Charge
        currents.extend([1.0] * 3)  # CC Charge
        currents.extend([0.5, 0.2, 0.1])  # CV Charge
        currents.extend([-1.0] * 10)  # Discharge
        currents.extend([0.0] * 4)  # Rest

    voltages = []
    for _ in range(2):
        voltages.extend([3.0, 3.3, 3.6])  # CC Charge
        voltages.extend([4.2, 4.2, 4.2])  # CV Charge
        voltages.extend([3.0, 3.3, 3.6])  # CC Charge
        voltages.extend([4.2, 4.2, 4.2])  # CV Charge
        voltages.extend([4.0, 3.7, 3.5, 3.2, 3.0, 4.0, 3.7, 3.5, 3.2, 3.0])
        voltages.extend([3.0] * 4)  # Rest

    dataframe = pl.DataFrame(
        {
            "Time [s]": list(range(len(steps))),
            "Step": steps,
            "Event": events,
            "Current [A]": currents,
            "Voltage [V]": voltages,
            "Capacity [Ah]": [0.0] * len(steps),
        },
    )
    step_descriptions = {
        "Step": [1, 2, 3, 4],
        "Description": ["CC Charge", "CV Charge", "Discharge", "Rest"],
    }

    # Outer cycle ends on Step 4; Inner cycle ends on Step 2
    cycle_info = [(1, 4, 2), (1, 2, 2)]
    return filters.Experiment(
        lf=dataframe,
        info={},
        step_descriptions=step_descriptions,
        cycle_info=cycle_info,
    )


def test_cycle_generic(generic_experiment):
    """Test cycle filtering on the generic synthetic experiment."""
    # Outer cycles (each 26 rows)
    assert generic_experiment.cycle(0).data["Time [s]"].to_list() == list(range(26))
    assert generic_experiment.cycle(1).data["Time [s]"].to_list() == list(range(26, 52))
    assert generic_experiment.cycle(-1).data["Time [s]"].to_list() == list(
        range(26, 52)
    )

    # Inner cycles
    next_cycle = generic_experiment.cycle(1)
    assert next_cycle.cycle_info == [(1, 2, 2)]
    # Inner cycle 0 of outer cycle 1: rows 26-31
    assert next_cycle.cycle(0).data["Time [s]"].to_list() == list(range(26, 32))
    # Inner cycle 1: rows 32-37
    assert next_cycle.cycle(1).data["Time [s]"].to_list() == list(range(32, 38))
    # Inner cycle 2 (remaining steps 3 and 4): rows 38-51
    assert next_cycle.cycle(2).data["Time [s]"].to_list() == list(range(38, 52))

    # Test inferred cycles
    generic_experiment.cycle_info = []
    # Step decreases from 2 to 1 at index 6, from 4 to 1 at index 26,
    # from 2 to 1 at index 32.
    # Step sequence: 1,1,1,2,2,2, | 1,1,1,2,2,2, | 3,3,3,3,3,3,3,3,3,3,4,4,4,4 | ...
    # Step decreases at indices 6, 26, 32.
    assert generic_experiment.cycle(0).data["Time [s]"].to_list() == list(range(6))
    assert generic_experiment.cycle(1).data["Time [s]"].to_list() == list(range(6, 26))
    assert generic_experiment.cycle(2).data["Time [s]"].to_list() == list(range(26, 32))
    assert generic_experiment.cycle(-1).data["Time [s]"].to_list() == list(
        range(32, 52)
    )


def test_cycle_chaining_generic(generic_experiment):
    """Test chained cycle filtering on nested cycle_info."""
    data = generic_experiment.cycle(1).cycle(2).data
    assert data["Time [s]"].to_list() == list(range(38, 52))


def test_step_generic(generic_experiment):
    """Test step filtering on the generic synthetic experiment."""
    # Step groups: 0:[1,1,1], 1:[2,2,2], 2:[1,1,1], 3:[2,2,2], 4:[3...], 5:[4...]
    # Cycle 2 starts at index 6: 6:[1,1,1], 7:[2,2,2], 8:[1,1,1], 9:[2,2,2],
    # 10:[3...], 11:[4...]

    data = generic_experiment.step(0).data
    assert data["Time [s]"].to_list() == [0, 1, 2]
    assert (data["Step"] == 1).all()

    # step(2) is the second occurrence of Step 1 (rows 6-8)
    data = generic_experiment.step(2).data
    assert data["Time [s]"].to_list() == [6, 7, 8]

    # step(6) is the first Step 1 in the second outer cycle (rows 26-28)
    data = generic_experiment.step(6).data
    assert data["Time [s]"].to_list() == [26, 27, 28]

    # Multiple steps
    data = generic_experiment.step(0, 1).data
    assert data["Time [s]"].to_list() == list(range(6))
    assert data["Step"].unique().to_list() == [1, 2]


def test_charge_generic(generic_experiment):
    """Test charge filtering on the generic synthetic experiment."""
    # Step 1 and 2 are charge.
    # Groups matching Current > 0:
    # 0:[0-2], 1:[3-5], 2:[6-8], 3:[9-11], 4:[26-28], 5:[29-31], 6:[32-34], 7:[35-37]

    # charge(0) should be event 0 (Step 1, rows 0-2)
    data = generic_experiment.charge(0).data
    assert data["Time [s]"].to_list() == [0, 1, 2]
    assert (data["Step"] == 1).all()

    # charge(1) should be event 1 (Step 2, rows 3-5)
    data = generic_experiment.charge(1).data
    assert data["Time [s]"].to_list() == [3, 4, 5]
    assert (data["Step"] == 2).all()

    # charge(4) should be first Step 1 in second outer cycle (rows 26-28)
    data = generic_experiment.charge(4).data
    assert data["Time [s]"].to_list() == [26, 27, 28]


def test_discharge_generic(generic_experiment):
    """Test discharge filtering on the generic synthetic experiment."""
    # Step 3 is discharge.
    # Groups matching Current < 0: 0:[12-21], 1:[38-47]
    data = generic_experiment.discharge(0).data
    assert data["Time [s]"].to_list() == list(range(12, 22))
    assert (data["Step"] == 3).all()
    assert (data["Current [A]"] < 0).all()

    data = generic_experiment.discharge(1).data
    assert data["Time [s]"].to_list() == list(range(38, 48))


def test_rest_generic(generic_experiment):
    """Test rest filtering on the generic synthetic experiment."""
    # Step 4 is rest.
    # Groups matching Current == 0: 0:[22-25], 1:[48-51]
    data = generic_experiment.rest(0).data
    assert data["Time [s]"].to_list() == list(range(22, 26))
    assert (data["Step"] == 4).all()
    assert (data["Current [A]"] == 0).all()


def test_constant_current_generic(generic_experiment):
    """Test constant_current filtering on the generic synthetic experiment."""
    data = generic_experiment.constant_current(0, target=1).data
    assert data["Time [s]"].to_list() == [0, 1, 2]
    assert (data["Current [A]"] == 1.0).all()

    data = generic_experiment.constant_current(0, target=-1).data
    assert data["Time [s]"].to_list() == list(range(12, 22))
    assert (data["Current [A]"] == -1.0).all()


def test_constant_voltage_generic(generic_experiment):
    """Test constant_voltage filtering on the generic synthetic experiment."""
    data = generic_experiment.constant_voltage(0, target=4.2).data
    assert data["Time [s]"].to_list() == [3, 4, 5]


def test_iter_filters_generic(generic_experiment):
    """Test iterator versions of filters on generic experiment."""
    # iter_step: 6 steps per outer cycle * 2 = 12
    steps = list(generic_experiment.iter_step())
    assert len(steps) == 12

    # iter_charge: Step 1 and 2 match Current > 0.
    # (Step 1, Step 2, Step 1, Step 2) * 2 = 8 groups
    charges = list(generic_experiment.iter_charge())
    assert len(charges) == 8

    # iter_discharge: Step 3 matches Current < 0.
    # Step 3 * 2 = 2 groups
    discharges = list(generic_experiment.iter_discharge())
    assert len(discharges) == 2

    # iter_rest: Step 4 matches Current == 0.
    # Step 4 * 2 = 2 groups
    rests = list(generic_experiment.iter_rest())
    assert len(rests) == 2

    # iter_chargeordischarge: Step 1, 2, 3 match abs(Current) > 0.
    # (Step 1, Step 2, Step 1, Step 2, Step 3) * 2 = 10 groups
    cods = list(generic_experiment.iter_chargeordischarge())
    assert len(cods) == 10


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
