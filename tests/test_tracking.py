import numpy as np
import pytest
from numba import njit
from numba.types import float32, float64

from omnisolver.pt.tracking import construct_tracker


def assert_records_agree(tracker, states, energies):
    recorded_states, recorded_energies = tracker.records()
    np.testing.assert_array_equal(states, recorded_states)
    np.testing.assert_array_equal(recorded_energies, energies)


class TestTrackersInitialization:
    @pytest.fixture(
        params=[
            (float32, 1),
            (float64, 1),
            (float32, 100),
            (float64, 100),
        ],
        scope="function",
    )
    def tracker(self, request):
        return construct_tracker(*request.param)

    def test_new_ground_only_tracker_retains_initial_configuration_as_the_best_one(self, tracker):
        initial_state = np.array([1, -1, 1, 1], dtype=np.int8)
        initial_energy = 5.0

        tracker.digest(initial_state, initial_energy)

        assert_records_agree(tracker, [initial_state], [initial_energy])

    def test_modifications_of_initial_state_from_python_code_are_not_reflected_in_tracker(
        self, tracker
    ):
        initial_state = np.array([1, -1, 1, 1], dtype=np.int8)
        initial_energy = 5.0
        expected_state = initial_state.copy()

        tracker.digest(initial_state, initial_energy)
        initial_state[0] = -1

        assert_records_agree(tracker, [expected_state], [initial_energy])

    def test_modifications_of_initial_state_from_jit_code_are_not_reflected_in_tracker(
        self, tracker
    ):
        initial_state = np.array([1, -1, 1, 1], dtype=np.int8)
        initial_energy = 5.0
        expected_state = initial_state.copy()

        def _foo(state, energy, tracker):
            tracker.digest(state, energy)
            state[0] = -1
            return tracker.records()

        recorded_states, recorded_energies = _foo(initial_state, initial_energy, tracker)
        np.testing.assert_array_equal([expected_state], recorded_states)
        assert recorded_energies == [initial_energy]


class TestGroundOnlyTracker:
    @pytest.fixture(scope="function")
    def input_example(self):
        rng = np.random.default_rng(42)
        energies = rng.integers(0, 10, size=(10)) / 2
        states = rng.integers(0, 1, size=(10, 8), dtype=np.int8) * 2 - 1
        best_idx = np.argmin(energies)
        best_state, best_energy = states[best_idx], energies[best_idx]
        return states, energies, best_state, best_energy, best_idx

    @pytest.fixture(params=[(float32, 1), (float64, 1)], scope="function")
    def tracker(self, request):
        return construct_tracker(*request.param)

    def test_digesting_new_configuration_with_lower_energy_updates_records(self, tracker):
        initial_state = np.array([1, -1, -1, 1, 1], dtype=np.int8)
        initial_energy = 2.1
        new_state = np.array([1, 1, -1, 1, -1], dtype=np.int8)
        new_energy = -0.5

        tracker.digest(initial_state, initial_energy)
        tracker.digest(new_state, new_energy)

        assert_records_agree(tracker, [new_state], [new_energy])

    def test_digesting_new_configuration_with_higher_energy_leaves_records_untouched(self, tracker):
        initial_state = np.array([1, -1, -1, 1, 1], dtype=np.int8)
        initial_energy = 2.5
        new_state = np.array([1, 1, -1, 1, -1], dtype=np.int8)
        new_energy = 3.0
        tracker.digest(initial_state, initial_energy)
        tracker.digest(new_state, new_energy)

        assert_records_agree(tracker, [initial_state], [initial_energy])

    def test_only_single_lowest_energy_state_is_recorded_after_successive_digestions(
        self, tracker, input_example
    ):
        states, energies, best_state, best_energy, _ = input_example

        for state, energy in zip(states, energies):
            tracker.digest(state, energy)

        assert_records_agree(tracker, [best_state], [best_energy])

    def test_stored_states_are_copied_in_python_code(self, tracker, input_example):
        states, energies, best_state, best_energy, _ = input_example
        expected_state = best_state.copy()
        tracker.digest(states[0], energies[0])

        for state, energy in zip(states[1:], energies[1:]):
            tracker.digest(state, energy)

        best_state[0] = -best_state[0]

        assert_records_agree(tracker, [expected_state], [best_energy])

    def test_stored_states_are_copied_in_jit_code(self, tracker, input_example):
        states, energies, best_state, best_energy, best_idx = input_example
        expected_state = best_state.copy()

        @njit
        def _foo(tracker, states, energies):
            for state, energy in zip(states, energies):
                tracker.digest(state, energy)

            states[best_idx][0] = -states[best_idx][0]

        _foo(tracker, states, energies)

        assert_records_agree(tracker, [expected_state], [best_energy])


@pytest.mark.parametrize("tracker", [(float32, 5), (float64, 5)], indirect=True)
class TestLowEnergySpectrumTracker:
    @pytest.fixture(
        params=[
            (float32, 100),
            (float64, 100),
        ]
    )
    def tracker(self, request):
        return construct_tracker(*request.param)

    def test_digesting_the_same_record_is_idempotent(self, tracker):
        initial_state = np.array([-1, 1, 1], dtype=np.int8)
        initial_energy = 0.5
        tracker.digest(initial_state, initial_energy)

        tracker.digest(initial_state, initial_energy)

        assert_records_agree(tracker, [initial_state], [initial_energy])

    def test_records_returns_best_sorted_states(self, tracker):
        states = [
            np.array([1, 1, 1], dtype=np.int8),
            np.array([-1, 1, 1], dtype=np.int8),
            np.array([-1, -1, 1], dtype=np.int8),
            np.array([1, -1, 1], dtype=np.int8),
            np.array([1, -1, -1], dtype=np.int8),
            np.array([-1, -1, -1], dtype=np.int8),
            np.array([-1, 1, -1], dtype=np.int8),
        ]
        energies = [-2.5, -2.0, 2.0, 1.5, -1.0, -0.5, 0.0, 3.0]

        for state, energy in zip(states, energies):
            tracker.digest(state, energy)

        assert_records_agree(
            tracker,
            [states[0], states[1], states[4], states[5], states[6]],
            [-2.5, -2.0, -1.0, -0.5, 0.0],
        )

    def test_stored_states_are_copied_in_python_code(self, tracker):
        initial_state = np.array([1, 1, 1], dtype=np.int8)
        initial_energy = -2.5
        new_state = np.array([1, 1, -1], dtype=np.int8)
        expected_state = new_state.copy()
        new_energy = -3.0
        tracker.digest(initial_state, initial_energy)

        tracker.digest(new_state, new_energy)
        new_state[0] = -1

        assert_records_agree(tracker, [expected_state, initial_state], [new_energy, initial_energy])

    def test_stored_states_are_copied_in_jit_code(self, tracker):
        initial_state = np.array([1, 1, 1], dtype=np.int8)
        initial_energy = -2.5
        new_state = np.array([1, 1, -1], dtype=np.int8)
        expected_state = new_state.copy()
        new_energy = -3.0

        @njit
        def _foo(tracker, state, energy):
            tracker.digest(state, energy)
            state[0] = -state[0]

        tracker.digest(initial_state, initial_energy)
        _foo(tracker, new_state, new_energy)

        assert_records_agree(tracker, [expected_state, initial_state], [new_energy, initial_energy])
