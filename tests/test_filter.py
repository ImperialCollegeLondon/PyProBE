"""Tests for the filter module."""

import numpy as np
import polars as pl
import pytest

import pyprobe.filters as filters
from pyprobe.filters import (
    _count_condition_groups,
    _extend_mask_with_preceding_point,
    _make_constant_condition,
    _make_group_marker_expr,
    _slice_to_mask_expr,
)


@pytest.fixture
def generic_experiment():
    """Return a synthetic experiment for testing nested cycles.

    Structure (per outer cycle, 26 rows):
      Step 1 x3, Step 2 x3, Step 1 x3, Step 2 x3, Step 3 x10, Step 4 x4
    Two outer cycles → 52 rows total.
    """
    outer_steps = [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3 + [3] * 10 + [4] * 4
    steps = outer_steps * 2

    events = []
    event_counts = [3, 3, 3, 3, 10, 4] * 2
    for i, count in enumerate(event_counts):
        events.extend([i] * count)

    currents = []
    for _ in range(2):
        currents.extend([1.0] * 3)
        currents.extend([0.5, 0.2, 0.1])
        currents.extend([1.0] * 3)
        currents.extend([0.5, 0.2, 0.1])
        currents.extend([-1.0] * 10)
        currents.extend([0.0] * 4)

    voltages = []
    for _ in range(2):
        voltages.extend([3.0, 3.3, 3.6])
        voltages.extend([4.2, 4.2, 4.2])
        voltages.extend([3.0, 3.3, 3.6])
        voltages.extend([4.2, 4.2, 4.2])
        voltages.extend([4.0, 3.7, 3.5, 3.2, 3.0, 4.0, 3.7, 3.5, 3.2, 3.0])
        voltages.extend([3.0] * 4)

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
    cycle_info = [(1, 4, 2), (1, 2, 2)]
    return filters.Experiment(
        lf=dataframe,
        info={},
        step_descriptions=step_descriptions,
        cycle_info=cycle_info,
    )


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


class TestExtendMaskWithPrecedingPoint:
    """Unit tests for _extend_mask_with_preceding_point."""

    @pytest.mark.parametrize(
        "input_mask, expected_mask",
        [
            (
                [False, False, True, True, False, False],
                [False, True, True, True, False, False],
            ),
            (
                [True, True, False, False, False, False],
                [True, True, False, False, False, False],
            ),
            ([False] * 6, [False] * 6),
            ([True] * 6, [True] * 6),
            (
                [False, True, False, False, True, False],
                [True, True, False, True, True, False],
            ),
        ],
    )
    def test_extend_mask_preceding_point_extends_true_runs(
        self, input_mask, expected_mask
    ):
        """Each contiguous True run gains the row before it."""
        df = pl.DataFrame({"mask": input_mask})
        result = df.select(_extend_mask_with_preceding_point(pl.col("mask")))
        assert result["mask"].to_list() == expected_mask


class TestMakeGroupMarkerExpr:
    """Unit tests for _make_group_marker_expr."""

    @pytest.mark.parametrize(
        "condition, expected_markers",
        [
            (
                pl.col("Current [A]") > 0,
                [True, False, False, False, False, False, False, False],
            ),
            (
                pl.col("Current [A]") < 0,
                [False, False, False, False, True, False, False, False],
            ),
            (
                pl.col("Current [A]") == 0,
                [False, False, True, False, False, False, True, False],
            ),
            (pl.lit(False), [False] * 8),
            (pl.lit(True), [True, False, True, False, True, False, True, False]),
        ],
    )
    def test_make_group_marker_marks_first_row_of_matching_groups(
        self, condition, expected_markers
    ):
        """Only the first row of each matching event group is marked True."""
        df = pl.DataFrame(
            {
                "Event": [0, 0, 1, 1, 2, 2, 3, 3],
                "Current [A]": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            },
        )
        marker = _make_group_marker_expr("Event", condition)
        result = df.select(marker.alias("marker"))
        assert result["marker"].to_list() == expected_markers


class TestCountConditionGroups:
    """Unit tests for _count_condition_groups."""

    @pytest.mark.parametrize(
        "condition, expected_count",
        [
            (None, 4),
            (pl.col("Current [A]") > 0, 1),
            (pl.col("Current [A]") == 0, 2),
            (pl.col("Current [A]") < 0, 1),
        ],
    )
    def test_count_condition_groups_counts_correctly(self, condition, expected_count):
        """Group count matches the number of distinct matching event runs."""
        df = pl.DataFrame(
            {
                "Event": [0, 0, 1, 1, 2, 2, 3, 3],
                "Current [A]": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            },
        )
        assert _count_condition_groups(df, "Event", condition) == expected_count
        assert _count_condition_groups(df.lazy(), "Event", condition) == expected_count


class TestSliceToMaskExpr:
    """Unit tests for _slice_to_mask_expr."""

    @pytest.mark.parametrize(
        "sl, expected_values",
        [
            (slice(2, 5), [2, 3, 4]),
            (slice(None, 3), [0, 1, 2]),
            (slice(7, None), [7, 8, 9]),
            (slice(-3, None), [7, 8, 9]),
            (slice(None, -2), [0, 1, 2, 3, 4, 5, 6, 7]),
            (slice(-5, -2), [5, 6, 7]),
            (slice(0, 10, 2), [0, 2, 4, 6, 8]),
            (slice(3, 0), []),
            (slice(-3, 0), [7, 8, 9]),
            (slice(-5, None, 2), [5, 7, 9]),
            (slice(-5, -1, 2), [5, 7]),
            (slice(None, None), list(range(10))),
        ],
    )
    def test_slice_to_mask_selects_correct_values(self, sl, expected_values):
        """Mask selects exactly the values implied by the slice."""
        df = pl.DataFrame({"Value": list(range(10))})
        asc_rank = pl.col("Value").rank("dense")
        desc_rank = pl.col("Value").rank("dense", descending=True)
        mask = _slice_to_mask_expr(sl, asc_rank, desc_rank)
        assert df.filter(mask)["Value"].to_list() == expected_values

    def test_slice_to_mask_invalid_args_raise_value_error(self):
        """Negative step or missing desc_rank for negative bounds raises ValueError."""
        asc_rank = pl.col("Value").rank("dense")

        with pytest.raises(ValueError, match="Negative step"):
            _slice_to_mask_expr(slice(0, 5, -1), asc_rank, None)

        with pytest.raises(
            ValueError, match="Negative slice start requires a descending rank"
        ):
            _slice_to_mask_expr(slice(-3, None), asc_rank, None)

        with pytest.raises(
            ValueError, match="Negative slice stop requires a descending rank"
        ):
            _slice_to_mask_expr(slice(None, -2), asc_rank, None)

        with pytest.raises(
            ValueError, match="Negative slice start requires a descending rank"
        ):
            _slice_to_mask_expr(slice(-2, None, 2), asc_rank, None)


class TestFilterBuildMask:
    """Unit tests for Filter._build_mask."""

    @pytest.mark.parametrize(
        "filt, indices, expected_events",
        [
            (filters._Filter("Event"), (), [0, 0, 1, 1, 2, 2, 3, 3]),
            (filters._Filter("Event"), (0,), [0, 0]),
            (filters._Filter("Event"), (-1,), [3, 3]),
            (filters._Filter("Event"), (range(0, 2),), [0, 0, 1, 1]),
            (filters._Filter("Event"), (slice(1, 3),), [1, 1, 2, 2]),
            (filters._Filter("Event", pl.col("Current [A]") > 0), (), [0, 0]),
            (filters._Filter("Event", pl.col("Current [A]") > 0), (0,), [0, 0]),
            (filters._Filter("Event", pl.col("Current [A]") == 0), (), [1, 1, 3, 3]),
            (filters._Filter("Event", pl.col("Current [A]") == 0), (0,), [1, 1]),
            (filters._Filter("Event", pl.col("Current [A]") == 0), (1,), [3, 3]),
            (filters._Filter("Event"), ("unsupported",), [0, 0, 1, 1, 2, 2, 3, 3]),
            (
                filters._Filter("Event", pl.col("Current [A]") > 0),
                ("unsupported",),
                [0, 0],
            ),
        ],
    )
    def test_filter_build_mask_selects_expected_events(
        self, filt, indices, expected_events
    ):
        """Mask selects rows whose Event values match the expected list."""
        df = pl.DataFrame(
            {
                "Event": [0, 0, 1, 1, 2, 2, 3, 3],
                "Current [A]": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            },
        )
        mask = filt._build_mask(indices)
        assert df.filter(mask)["Event"].to_list() == expected_events


class TestFilterExpandPositions:
    """Unit tests for Filter._expand_positions."""

    @pytest.mark.parametrize(
        "indices, expected_positions",
        [
            ((), [0, 1, 2, 3]),
            ((0, 1), [0, 1]),
            ((slice(-2, None),), [2, 3]),
        ],
    )
    def test_filter_expand_positions_resolves_to_integers(
        self, indices, expected_positions
    ):
        """Positional selectors expand to a flat list of zero-based integers."""
        df = pl.DataFrame(
            {
                "Event": [0, 0, 1, 1, 2, 2, 3, 3],
                "Current [A]": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
            },
        )
        lf = df.lazy()
        f_event = filters._Filter("Event")
        assert f_event._expand_positions(lf, indices) == expected_positions

    def test_filter_expand_positions_invalid_index_raises(self):
        """Negative step slice and unsupported index types raise the right errors."""
        df = pl.DataFrame({"Event": [0, 1]})
        lf = df.lazy()
        f_event = filters._Filter("Event")

        with pytest.raises(ValueError, match="Negative step"):
            f_event._expand_positions(lf, (slice(0, None, -1),))

        with pytest.raises(TypeError, match="Unsupported index type"):
            f_event._expand_positions(lf, ("bad",))


class TestMakeConstantCondition:
    """Unit tests for _make_constant_condition."""

    @pytest.mark.parametrize(
        "target, rtol, values, expected_selected",
        [
            (1.0, 0.001, [1.0, 1.0005, 1.002, 2.0], [1.0, 1.0005]),
            (-1.0, 0.001, [-1.0, -1.0005, 1.0], [-1.0, -1.0005]),
            (1.0, 0.01, [1.0, 1.005, 1.015], [1.0, 1.005]),
        ],
    )
    def test_make_constant_condition_with_target_selects_band(
        self, target, rtol, values, expected_selected
    ):
        """Only rows within target ± |target|*rtol are selected."""
        df = pl.DataFrame({"col": values})
        cond = _make_constant_condition("col", target=target, rtol=rtol)
        assert df.filter(cond)["col"].to_list() == expected_selected

    def test_make_constant_condition_no_target_uses_mode(self):
        """Without a target, the global mode value is used as the target."""
        df = pl.DataFrame({"col": [1.0, 1.0, 1.0, 2.0]})
        cond = _make_constant_condition("col", rtol=0.001)
        assert df.filter(cond)["col"].to_list() == [1.0, 1.0, 1.0]

    def test_make_constant_condition_with_mask_filters_before_mode(self):
        """The mask pre-filters rows before the mode is computed."""
        df = pl.DataFrame({"col": [0.0, 0.0, 0.0, 1.0, 1.0, 1.001]})
        mask = pl.col("col") != 0
        cond = _make_constant_condition("col", rtol=0.001, mask=mask)
        assert df.filter(cond)["col"].to_list() == [1.0, 1.0, 1.001]


class TestStepAndCycleFiltering:
    """Integration tests for step() and cycle() using the BreakinCycles fixture."""

    def test_step_returns_correct_step(self, BreakinCycles_fixture, benchmark):
        """step(1) in cycle 0 returns Step 5."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).step(1).data)
        assert (data["Step"] == 5).all()

    def test_multi_step_returns_multiple_steps(self, BreakinCycles_fixture, benchmark):
        """step(range(1, 4)) returns Steps 5, 6, 7."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).step(range(1, 4)).data)
        assert (data["Step"].unique() == [5, 6, 7]).all()

    def test_cycle_returns_correct_cycle(self, BreakinCycles_fixture, benchmark):
        """cycle(2) returns only cycle 2 data with zeroed time and capacity."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(2).data)
        assert (data["Cycle"] == 2).all()
        assert (data["Step"].unique() == [4, 5, 6, 7]).all()
        assert data["Cycle Time [s]"][0] == 0
        assert data["Cycle Capacity [Ah]"][0] == 0

    def test_negative_cycle_index_selects_last(self, BreakinCycles_fixture, benchmark):
        """cycle(-1) returns the last cycle."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(-1).data)
        assert (data["Cycle"] == 4).all()
        assert (data["Step"].unique() == [4, 5, 6, 7]).all()

    def test_negative_step_index_selects_last(self, BreakinCycles_fixture, benchmark):
        """step(-1) returns the last step event in the cycle."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).step(-1).data)
        assert (data["Step"] == 7).all()

    def test_all_steps_returns_full_cycle(self, BreakinCycles_fixture, benchmark):
        """step() with no index returns all step events."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).step().data)
        assert (data["Cycle"] == 0).all()
        assert (data["Step"].unique() == [4, 5, 6, 7]).all()

    def test_zeroed_columns_reset_at_each_level(self, BreakinCycles_fixture):
        """Time and Capacity columns are zeroed at each filtering level."""
        exp = BreakinCycles_fixture
        cycle = exp.cycle(0)
        step = cycle.step(0)
        assert exp.get("Experiment Time [s]")[0] == 0
        assert exp.get("Experiment Capacity [Ah]")[0] == 0
        assert cycle.get("Cycle Time [s]")[0] == 0
        assert cycle.get("Cycle Capacity [Ah]")[0] == 0
        assert step.get("Step Time [s]")[0] == 0
        assert step.get("Step Capacity [Ah]")[0] == 0

    def test_multiple_indices_selects_union(self, BreakinCycles_fixture):
        """Multiple integer indices select the union of those step events."""
        data = BreakinCycles_fixture.cycle(0).step(0, 1, 2).data
        assert sorted(data["Step"].unique().to_list()) == [4, 5, 6]

    def test_multiple_indices_range_and_slice_combined(self, BreakinCycles_fixture):
        """Combining an integer and a slice index selects their union."""
        data = BreakinCycles_fixture.cycle(0).step(0, slice(2, 4)).data
        assert set(data["Step"].unique().to_list()) == {4, 6, 7}


class TestChargeDischargeRest:
    """Integration tests for charge(), discharge(), rest(), chargeordischarge()."""

    def test_charge_returns_positive_current(self, BreakinCycles_fixture, benchmark):
        """charge(0) returns only positive-current rows."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).charge(0).data)
        assert (data["Step"] == 6).all()
        assert (data["Current [A]"] > 0).all()

    def test_discharge_returns_negative_current(self, BreakinCycles_fixture, benchmark):
        """discharge(0) returns only negative-current rows."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).discharge(0).data)
        assert (data["Step"] == 4).all()
        assert (data["Current [A]"] < 0).all()
        with pytest.raises(ValueError):
            BreakinCycles_fixture.cycle(6).data

    def test_chargeordischarge_returns_nonzero_current(
        self, BreakinCycles_fixture, benchmark
    ):
        """Chargeordischarge selects both charge and discharge events by index."""
        data = benchmark(
            lambda: BreakinCycles_fixture.cycle(0).chargeordischarge(0).data
        )
        assert (data["Step"] == 4).all()
        assert (data["Current [A]"] < 0).all()
        data = BreakinCycles_fixture.cycle(0).chargeordischarge(1).data
        assert (data["Step"] == 6).all()
        assert (data["Current [A]"] > 0).all()

    def test_rest_returns_zero_current(self, BreakinCycles_fixture, benchmark):
        """rest(0) and rest(1) return zero-current rows at the expected steps."""
        data = benchmark(lambda: BreakinCycles_fixture.cycle(0).rest(0).data)
        assert (data["Step"] == 5).all()
        assert (data["Current [A]"] == 0).all()
        data = BreakinCycles_fixture.cycle(0).rest(1).data
        assert (data["Step"] == 7).all()
        assert (data["Current [A]"] == 0).all()


class TestConstantFilters:
    """Integration tests for constant_current() and constant_voltage()."""

    def test_constant_current_returns_target_rows(
        self, BreakinCycles_fixture, benchmark
    ):
        """constant_current(target=0.004) returns rows within the tolerance band."""
        data = benchmark(
            lambda: BreakinCycles_fixture.constant_current(1, target=0.004).data
        )
        assert np.isclose(data["Current [A]"].to_numpy().mean(), 0.004, rtol=0.001)
        assert data["Current [A]"].min() > 0.004 - 0.004 * 0.001
        assert data["Current [A]"].max() < 0.004 + 0.004 * 0.001

    def test_constant_voltage_returns_target_rows(
        self, BreakinCycles_fixture, benchmark
    ):
        """constant_voltage(target=4.2) returns rows within the tolerance band."""
        data = benchmark(
            lambda: BreakinCycles_fixture.constant_voltage(1, target=4.2).data
        )
        assert np.isclose(data["Voltage [V]"].to_numpy().mean(), 4.2, rtol=0.001)
        assert data["Voltage [V]"].min() > 4.195
        assert data["Voltage [V]"].max() < 4.2


class TestSlicing:
    """Integration tests for slice-based index selection."""

    def test_slice_positive_bounds_selects_range(self, BreakinCycles_fixture):
        """slice(0, 2) selects the first two step events."""
        data = BreakinCycles_fixture.cycle(0).step(slice(0, 2)).data
        assert sorted(data["Step"].unique().to_list()) == [4, 5]

    def test_slice_negative_bounds_selects_from_end(self, BreakinCycles_fixture):
        """slice(-2, None) selects the last two step events."""
        data = BreakinCycles_fixture.cycle(0).step(slice(-2, None)).data
        assert sorted(data["Step"].unique().to_list()) == [6, 7]

    def test_slice_mixed_bounds_selects_intersection(self, BreakinCycles_fixture):
        """slice(-3, 3) satisfies both negative start and positive stop bounds."""
        data = BreakinCycles_fixture.cycle(0).step(slice(-3, 3)).data
        assert sorted(data["Step"].unique().to_list()) == [5, 6]

    def test_slice_step_greater_than_one_strides(self, BreakinCycles_fixture):
        """slice(0, None, 2) selects every other step event."""
        data = BreakinCycles_fixture.cycle(0).step(slice(0, None, 2)).data
        assert sorted(data["Step"].unique().to_list()) == [4, 6]

    def test_slice_negative_start_open_ended(self, BreakinCycles_fixture):
        """slice(-2, None) selects the last two step events."""
        data = BreakinCycles_fixture.cycle(0).step(slice(-2, None)).data
        assert sorted(data["Step"].unique().to_list()) == [6, 7]

    def test_slice_negative_start_with_step_greater_than_one(
        self, BreakinCycles_fixture
    ):
        """slice(-4, None, 2) strides from the fourth-to-last event."""
        data = BreakinCycles_fixture.cycle(0).step(slice(-4, None, 2)).data
        assert sorted(data["Step"].unique().to_list()) == [4, 6]

    def test_slice_zero_stop_returns_empty(self, BreakinCycles_fixture):
        """slice(0, 0) produces an empty result."""
        with pytest.raises(ValueError, match="No data exists for this filter"):
            BreakinCycles_fixture.cycle(0).step(slice(0, 0)).data

    def test_slice_positive_start_zero_stop_returns_empty(self, BreakinCycles_fixture):
        """slice(1, 0) produces an empty result."""
        with pytest.raises(ValueError, match="No data exists for this filter"):
            BreakinCycles_fixture.cycle(0).step(slice(1, 0)).data

    def test_slice_negative_step_raises_error(self, BreakinCycles_fixture):
        """A negative step value raises ValueError."""
        with pytest.raises(ValueError, match="Negative step is not supported"):
            BreakinCycles_fixture.cycle(0).step(slice(3, 0, -1)).data


class TestIncludePrecedingPoint:
    """Tests for include_preceding_point across all singular filter methods."""

    def _assert_preceding_row_prepended(self, data_without, data_with, full_data):
        assert len(data_with) == len(data_without) + 1
        match = (
            (full_data["Time [s]"] == data_without["Time [s]"][0])
            & (full_data["Step"] == data_without["Step"][0])
            & (full_data["Event"] == data_without["Event"][0])
        )
        idx = int(np.where(match)[0][0])
        assert idx > 0
        assert data_with["Time [s]"][0] == full_data["Time [s]"][idx - 1]

    def test_step_prepends_preceding_row(self, BreakinCycles_fixture):
        """Step with include_preceding_point prepends exactly one row."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        data_without = cycle0.step(1).data
        data_with = cycle0.step(1, include_preceding_point=True).data
        self._assert_preceding_row_prepended(data_without, data_with, cycle0.data)
        assert (
            data_with.tail(len(data_without))["Time [s]"].to_list()
            == data_without["Time [s]"].to_list()
        )

    def test_charge_prepends_preceding_row(self, BreakinCycles_fixture):
        """Charge with include_preceding_point prepends exactly one row."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        data_without = cycle0.charge(0).data
        data_with = cycle0.charge(0, include_preceding_point=True).data
        self._assert_preceding_row_prepended(data_without, data_with, cycle0.data)

    def test_discharge_prepends_preceding_row(self, BreakinCycles_fixture):
        """Discharge with include_preceding_point prepends exactly one row."""
        data_without = BreakinCycles_fixture.discharge(1).data
        data_with = BreakinCycles_fixture.discharge(
            1, include_preceding_point=True
        ).data
        self._assert_preceding_row_prepended(
            data_without, data_with, BreakinCycles_fixture.data
        )

    def test_rest_prepends_preceding_row(self, BreakinCycles_fixture):
        """Rest with include_preceding_point prepends exactly one row."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        data_without = cycle0.rest(0).data
        data_with = cycle0.rest(0, include_preceding_point=True).data
        self._assert_preceding_row_prepended(data_without, data_with, cycle0.data)

    def test_chargeordischarge_prepends_preceding_row(self, BreakinCycles_fixture):
        """Chargeordischarge with include_preceding_point prepends exactly one row."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        data_without = cycle0.chargeordischarge(1).data
        data_with = cycle0.chargeordischarge(1, include_preceding_point=True).data
        self._assert_preceding_row_prepended(data_without, data_with, cycle0.data)

    def test_cycle_prepends_preceding_row(self, BreakinCycles_fixture):
        """Cycle with include_preceding_point prepends exactly one row."""
        data_without = BreakinCycles_fixture.cycle(1).data
        data_with = BreakinCycles_fixture.cycle(1, include_preceding_point=True).data
        self._assert_preceding_row_prepended(
            data_without, data_with, BreakinCycles_fixture.data
        )

    @pytest.mark.parametrize("exp_name", ["Break-in Cycles", "Discharge Pulses"])
    def test_procedure_experiment_prepends_preceding_row(
        self, procedure_fixture, exp_name
    ):
        """Procedure.experiment with include_preceding_point prepends one row."""
        data_without = procedure_fixture.experiment(exp_name).data
        data_with = procedure_fixture.experiment(
            exp_name, include_preceding_point=True
        ).data
        self._assert_preceding_row_prepended(
            data_without, data_with, procedure_fixture.data
        )

    def test_constant_current_prepends_preceding_row(self, generic_experiment):
        """constant_current with include_preceding_point prepends exactly one row."""
        data_without = generic_experiment.constant_current(1, target=1.0).data
        data_with = generic_experiment.constant_current(
            1, target=1.0, include_preceding_point=True
        ).data
        full_data = generic_experiment.data
        assert len(data_with) == len(data_without) + 1
        match = (full_data["Time [s]"] == data_without["Time [s]"][0]) & (
            full_data["Event"] == data_without["Event"][0]
        )
        idx = int(np.where(match)[0][0])
        assert idx > 0
        assert data_with["Time [s]"][0] == full_data["Time [s]"][idx - 1]

    def test_constant_voltage_prepends_preceding_row(self, generic_experiment):
        """constant_voltage with include_preceding_point prepends exactly one row."""
        data_without = generic_experiment.constant_voltage(1, target=4.2).data
        data_with = generic_experiment.constant_voltage(
            1, target=4.2, include_preceding_point=True
        ).data
        full_data = generic_experiment.data
        assert len(data_with) == len(data_without) + 1
        match = (full_data["Time [s]"] == data_without["Time [s]"][0]) & (
            full_data["Event"] == data_without["Event"][0]
        )
        idx = int(np.where(match)[0][0])
        assert idx > 0
        assert data_with["Time [s]"][0] == full_data["Time [s]"][idx - 1]


class TestIterators:
    """Tests for iter_* methods and their include_preceding_point behaviour."""

    def _assert_iter_preceding_row(self, results_without, results_with, full_data):
        assert len(results_with) == len(results_without)
        assert len(results_with[0].data) == len(results_without[0].data) + 1
        first_row = results_without[0].data
        match = (
            (full_data["Time [s]"] == first_row["Time [s]"][0])
            & (full_data["Step"] == first_row["Step"][0])
            & (full_data["Event"] == first_row["Event"][0])
        )
        idx = int(np.where(match)[0][0])
        assert idx > 0
        assert results_with[0].data["Time [s]"][0] == full_data["Time [s]"][idx - 1]

    def test_iter_step_single_index(self, BreakinCycles_fixture):
        """iter_step(0) yields one result containing step event 0."""
        results = list(BreakinCycles_fixture.cycle(0).iter_step(0))
        assert len(results) == 1
        assert results[0].data["Step"].unique().to_list() == [4]

    def test_iter_step_range(self, BreakinCycles_fixture):
        """iter_step(range(2)) yields two results, one per step event."""
        results = list(BreakinCycles_fixture.cycle(0).iter_step(range(2)))
        assert len(results) == 2
        assert results[0].data["Step"].unique().to_list() == [4]
        assert results[1].data["Step"].unique().to_list() == [5]

    def test_iter_step_slice(self, BreakinCycles_fixture):
        """iter_step(slice(0, 2)) yields one result per selected step event."""
        results = list(BreakinCycles_fixture.cycle(0).iter_step(slice(0, 2)))
        assert len(results) == 2
        assert results[0].data["Step"].unique().to_list() == [4]
        assert results[1].data["Step"].unique().to_list() == [5]

    def test_iter_charge_yields_all_groups(self, BreakinCycles_fixture):
        """iter_charge() yields one result per charge event."""
        results = list(BreakinCycles_fixture.iter_charge())
        assert len(results) == 5
        for item in results:
            assert item.data["Step"].unique().to_list() == [6]
            assert (item.data["Current [A]"] > 0).all()

    def test_iter_discharge_yields_all_groups(self, BreakinCycles_fixture):
        """iter_discharge() yields one result per discharge event."""
        results = list(BreakinCycles_fixture.iter_discharge())
        assert len(results) == 5
        for item in results:
            assert item.data["Step"].unique().to_list() == [4]
            assert (item.data["Current [A]"] < 0).all()

    def test_iter_rest_yields_all_groups(self, BreakinCycles_fixture):
        """iter_rest() yields one result per rest event."""
        results = list(BreakinCycles_fixture.iter_rest())
        assert len(results) == 10
        for item in results:
            assert (item.data["Current [A]"] == 0).all()

    def test_iter_chargeordischarge_yields_all_groups(self, BreakinCycles_fixture):
        """iter_chargeordischarge() yields one result per non-rest event."""
        results = list(BreakinCycles_fixture.iter_chargeordischarge())
        assert len(results) == 10
        for item in results:
            assert (item.data["Current [A]"].abs() > 0).all()

    def test_iter_cycle_yields_all_cycles(self, BreakinCycles_fixture):
        """iter_cycle() yields one result per cycle in order."""
        results = list(BreakinCycles_fixture.iter_cycle())
        assert len(results) == 5
        assert [c.data["Cycle"].unique()[0] for c in results] == [0, 1, 2, 3, 4]

    def test_iter_step_with_include_preceding_point(self, BreakinCycles_fixture):
        """iter_step with include_preceding_point adds one row to each result."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        results_without = list(cycle0.iter_step(slice(1, 2)))
        results_with = list(cycle0.iter_step(slice(1, 2), include_preceding_point=True))
        self._assert_iter_preceding_row(results_without, results_with, cycle0.data)

    def test_iter_charge_with_include_preceding_point(self, BreakinCycles_fixture):
        """iter_charge with include_preceding_point adds one row to each result."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        results_without = list(cycle0.iter_charge(slice(0, 1)))
        results_with = list(
            cycle0.iter_charge(slice(0, 1), include_preceding_point=True)
        )
        self._assert_iter_preceding_row(results_without, results_with, cycle0.data)

    def test_iter_discharge_with_include_preceding_point(self, BreakinCycles_fixture):
        """iter_discharge with include_preceding_point adds one row to each result."""
        results_without = list(BreakinCycles_fixture.iter_discharge(slice(1, 2)))
        results_with = list(
            BreakinCycles_fixture.iter_discharge(
                slice(1, 2), include_preceding_point=True
            )
        )
        self._assert_iter_preceding_row(
            results_without, results_with, BreakinCycles_fixture.data
        )

    def test_iter_rest_with_include_preceding_point(self, BreakinCycles_fixture):
        """iter_rest with include_preceding_point adds one row to each result."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        results_without = list(cycle0.iter_rest(slice(0, 1)))
        results_with = list(cycle0.iter_rest(slice(0, 1), include_preceding_point=True))
        self._assert_iter_preceding_row(results_without, results_with, cycle0.data)

    def test_iter_chargeordischarge_with_include_preceding_point(
        self, BreakinCycles_fixture
    ):
        """iter_chargeordischarge with include_preceding_point adds one row each."""
        cycle0 = BreakinCycles_fixture.cycle(0)
        results_without = list(cycle0.iter_chargeordischarge(slice(1, 2)))
        results_with = list(
            cycle0.iter_chargeordischarge(slice(1, 2), include_preceding_point=True)
        )
        self._assert_iter_preceding_row(results_without, results_with, cycle0.data)

    def test_iter_cycle_with_include_preceding_point(self, BreakinCycles_fixture):
        """iter_cycle with include_preceding_point adds one row to each result."""
        results_without = list(BreakinCycles_fixture.iter_cycle(slice(1, 2)))
        results_with = list(
            BreakinCycles_fixture.iter_cycle(slice(1, 2), include_preceding_point=True)
        )
        self._assert_iter_preceding_row(
            results_without, results_with, BreakinCycles_fixture.data
        )

    def test_iter_constant_current_yields_all_groups(self, generic_experiment):
        """iter_constant_current yields one result per matching current group."""
        results = list(generic_experiment.iter_constant_current(target=1.0))
        assert len(results) == 4
        for r in results:
            assert (r.data["Current [A]"] == 1.0).all()

        results_neg = list(generic_experiment.iter_constant_current(target=-1.0))
        assert len(results_neg) == 2
        for r in results_neg:
            assert (r.data["Current [A]"] == -1.0).all()

    def test_iter_constant_voltage_yields_all_groups(self, generic_experiment):
        """iter_constant_voltage yields one result per matching voltage group."""
        results = list(generic_experiment.iter_constant_voltage(target=4.2))
        assert len(results) == 4
        for r in results:
            assert (r.data["Voltage [V]"] == 4.2).all()


class TestGenericExperiment:
    """Integration tests using the synthetic generic_experiment fixture."""

    def test_cycle_filters_outer_and_inner_cycles(self, generic_experiment):
        """Outer and inner cycle filtering both return the correct row ranges."""
        assert generic_experiment.cycle(0).data["Time [s]"].to_list() == list(range(26))
        assert generic_experiment.cycle(1).data["Time [s]"].to_list() == list(
            range(26, 52)
        )
        assert generic_experiment.cycle(-1).data["Time [s]"].to_list() == list(
            range(26, 52)
        )

        next_cycle = generic_experiment.cycle(1)
        assert next_cycle.cycle_info == [(1, 2, 2)]
        assert next_cycle.cycle(0).data["Time [s]"].to_list() == list(range(26, 32))
        assert next_cycle.cycle(1).data["Time [s]"].to_list() == list(range(32, 38))
        assert next_cycle.cycle(2).data["Time [s]"].to_list() == list(range(38, 52))

    def test_cycle_inferred_from_step_decrease(self, generic_experiment):
        """When cycle_info is empty, cycles are inferred from step number decreases."""
        generic_experiment.cycle_info = []
        assert generic_experiment.cycle(0).data["Time [s]"].to_list() == list(range(6))
        assert generic_experiment.cycle(1).data["Time [s]"].to_list() == list(
            range(6, 26)
        )
        assert generic_experiment.cycle(2).data["Time [s]"].to_list() == list(
            range(26, 32)
        )
        assert generic_experiment.cycle(-1).data["Time [s]"].to_list() == list(
            range(32, 52)
        )

    def test_cycle_chaining_selects_correct_rows(self, generic_experiment):
        """Chained cycle() calls filter outer then inner cycles correctly."""
        data = generic_experiment.cycle(1).cycle(2).data
        assert data["Time [s]"].to_list() == list(range(38, 52))

    def test_step_selects_by_event_group(self, generic_experiment):
        """step() selects rows by event group index."""
        data = generic_experiment.step(0).data
        assert data["Time [s]"].to_list() == [0, 1, 2]
        assert (data["Step"] == 1).all()

        data = generic_experiment.step(2).data
        assert data["Time [s]"].to_list() == [6, 7, 8]

        data = generic_experiment.step(6).data
        assert data["Time [s]"].to_list() == [26, 27, 28]

        data = generic_experiment.step(0, 1).data
        assert data["Time [s]"].to_list() == list(range(6))

    def test_charge_selects_positive_current_groups(self, generic_experiment):
        """charge() selects groups by positive-current event index."""
        data = generic_experiment.charge(0).data
        assert data["Time [s]"].to_list() == [0, 1, 2]
        assert (data["Step"] == 1).all()

        data = generic_experiment.charge(1).data
        assert data["Time [s]"].to_list() == [3, 4, 5]
        assert (data["Step"] == 2).all()

        data = generic_experiment.charge(4).data
        assert data["Time [s]"].to_list() == [26, 27, 28]

    def test_discharge_selects_negative_current_groups(self, generic_experiment):
        """discharge() selects groups by negative-current event index."""
        data = generic_experiment.discharge(0).data
        assert data["Time [s]"].to_list() == list(range(12, 22))
        assert (data["Step"] == 3).all()
        assert (data["Current [A]"] < 0).all()

        data = generic_experiment.discharge(1).data
        assert data["Time [s]"].to_list() == list(range(38, 48))

    def test_rest_selects_zero_current_groups(self, generic_experiment):
        """rest() selects groups where current is zero."""
        data = generic_experiment.rest(0).data
        assert data["Time [s]"].to_list() == list(range(22, 26))
        assert (data["Step"] == 4).all()
        assert (data["Current [A]"] == 0).all()

    def test_constant_current_selects_by_target(self, generic_experiment):
        """constant_current(target) selects groups near the specified current."""
        data = generic_experiment.constant_current(0, target=1).data
        assert data["Time [s]"].to_list() == [0, 1, 2]
        assert (data["Current [A]"] == 1.0).all()

        data = generic_experiment.constant_current(0, target=-1).data
        assert data["Time [s]"].to_list() == list(range(12, 22))
        assert (data["Current [A]"] == -1.0).all()

    def test_constant_voltage_selects_by_target(self, generic_experiment):
        """constant_voltage(target) selects groups near the specified voltage."""
        data = generic_experiment.constant_voltage(0, target=4.2).data
        assert data["Time [s]"].to_list() == [3, 4, 5]

    def test_iter_filters_yield_correct_group_counts(self, generic_experiment):
        """All iter_* methods yield the expected number of groups."""
        assert len(list(generic_experiment.iter_step())) == 12
        assert len(list(generic_experiment.iter_charge())) == 8
        assert len(list(generic_experiment.iter_discharge())) == 2
        assert len(list(generic_experiment.iter_rest())) == 2
        assert len(list(generic_experiment.iter_chargeordischarge())) == 10


class TestParametricConstantFilters:
    """Tests for constant_current/constant_voltage with explicit target and rtol."""

    def test_constant_voltage_with_target_excludes_other_level(self, multilevel_cv):
        """Only the 4.2 V hold is returned; negative target matches nothing."""
        data = multilevel_cv.constant_voltage(0, target=4.2, rtol=0.001).data
        assert data["Voltage [V]"].min() >= 4.2 * 0.999
        assert data["Voltage [V]"].max() <= 4.2 * 1.001
        assert len(data) == 100
        assert multilevel_cv.constant_voltage(target=-4.2).lf.collect().is_empty()

    def test_constant_current_with_target_excludes_other_level(self, multilevel_cc):
        """Only the 1.0 A hold is returned; negative target matches nothing."""
        data = multilevel_cc.constant_current(0, target=1.0, rtol=0.001).data
        assert data["Current [A]"].min() >= 1.0 * 0.999
        assert data["Current [A]"].max() <= 1.0 * 1.001
        assert len(data) == 100
        assert multilevel_cc.constant_current(target=-1.0).lf.collect().is_empty()

    def test_constant_voltage_no_target_uses_dominant_level(self, multilevel_cv):
        """Without a target, both indexed groups return the dominant (4.2 V) level."""
        data_0 = multilevel_cv.constant_voltage(0).data
        data_1 = multilevel_cv.constant_voltage(1).data
        assert np.isclose(data_0["Voltage [V]"].mean(), 4.2, rtol=0.001)
        assert np.isclose(data_1["Voltage [V]"].mean(), 4.2, rtol=0.001)
        with pytest.raises(ValueError):
            multilevel_cv.constant_voltage(2).data

    def test_constant_voltage_rtol_controls_band_width(self):
        """Wider rtol includes more rows; narrower rtol excludes borderline rows."""
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
        wide = exp.constant_voltage(target=4.2, rtol=0.003).data
        narrow = exp.constant_voltage(target=4.2, rtol=0.001).data
        assert len(wide) == 20
        assert len(narrow) == 10
        assert narrow["Voltage [V]"].min() == 4.2

    def test_constant_voltage_index_with_target_selects_first_group(
        self, multilevel_cv
    ):
        """constant_voltage(0, target=X) returns only the first matching hold."""
        first = multilevel_cv.constant_voltage(0, target=4.2, rtol=0.001).data
        both = multilevel_cv.constant_voltage(target=4.2, rtol=0.001).data
        assert len(first) < len(both)
        assert len(first) == 100
        assert first["Voltage [V]"].min() >= 4.2 * 0.999
