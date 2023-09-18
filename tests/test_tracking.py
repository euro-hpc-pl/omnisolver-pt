import numpy as np
import pytest
from numba.types import float32, float64

from omnisolver.pt.tracking import construct_tracker


class TestTrackersInitialization:
    @pytest.mark.parametrize(
        "tracker_cls", [construct_tracker(float32, 1), construct_tracker(float64, 1)]
    )
    def test_new_ground_only_tracker_retains_initial_configuration_as_the_best_one(
        self, tracker_cls
    ):
        initial_state = np.array([1, -1, 1, 1], dtype=np.int8)
        initial_energy = 5.0

        tracker = tracker_cls(initial_state, initial_energy)

        recorded_states, recorded_energies = tracker.records()
        np.testing.assert_array_equal([initial_state], recorded_states)
        assert recorded_energies == [initial_energy]


@pytest.mark.parametrize(
    "tracker_cls", [construct_tracker(float32, 1), construct_tracker(float64, 1)]
)
class TestGroundOnlyTracker:
    def test_storing_new_configuration_with_lower_energy_updates_records(self, tracker_cls):
        initial_state = np.array([1, -1, -1, 1, 1], dtype=np.int8)
        initial_energy = 2.1
        new_state = np.array([1, 1, -1, 1, -1], dtype=np.int8)
        new_energy = -0.5
        tracker = tracker_cls(initial_state, initial_energy)

        tracker.store(new_state, new_energy)

        recorded_states, recorded_energies = tracker.records()
        np.testing.assert_array_equal([new_state], recorded_states)
        assert recorded_energies == [new_energy]

    def test_storing_new_configuration_with_higher_energy_leaves_records_untouched(
        self, tracker_cls
    ):
        initial_state = np.array([1, -1, -1, 1, 1], dtype=np.int8)
        initial_energy = 2.5
        new_state = np.array([1, 1, -1, 1, -1], dtype=np.int8)
        new_energy = 3.0
        tracker = tracker_cls(initial_state, initial_energy)

        tracker.store(new_state, new_energy)

        recorded_states, recorded_energies = tracker.records()
        np.testing.assert_array_equal([initial_state], recorded_states)
        assert recorded_energies == [initial_energy]

    def test_only_single_lowest_energy_state_is_recorded_after_successive_stores(self, tracker_cls):
        rng = np.random.default_rng(42)
        energies = rng.integers(0, 10, size=(10)) / 2
        states = rng.integers(-1, 1, size=(10, 8), dtype=np.int8)
        idx = np.argmin(energies)
        best_state, best_energy = states[idx], energies[idx]
        tracker = tracker_cls(states[0], energies[0])

        for state, energy in zip(states[1:], energies[1:]):
            tracker.store(state, energy)

        recorded_states, recorded_energies = tracker.records()
        np.testing.assert_array_equal([best_state], recorded_states)
        assert recorded_energies == [best_energy]
