"""Module providing several tracker implementations.

A tracker is an object used by replica for keeping track of relevant information. For the replica,
the tracker objects are opaque, meaning that the replica only feeds the newly encountered
configurations and is not concerned at all about what the tracker does with them. This approach
one to decouple Replicas and Trackers.

Currently, two forms of tracking are implemented:
- Tracking only the best encountered configuration.
- Tracking up to M best encountered configurations, where M is a user supplied parmeter.

At a first glance, the first option may seem like a special case of the second one.
However, we distinguish it from the second one as it has significantly more performant,
specialized implementation.

To keep the user away from having to manually construct trackers, this module provides a
simple function `tracker factory(dtype, num_states)`, which constructs an appropriate
tracker given the dtype of the model and desired number of states to store.
"""
from functools import lru_cache
from heapq import heappop, heappush
from typing import Protocol, Sequence, Tuple

import numpy as np
from numba.experimental import jitclass
from numba.types import List, int8, int64

from ._numba_helpers import numba_type_of_cls

Records = Tuple[Sequence[np.ndarray], Sequence[float]]


class Tracker(Protocol):
    """Protocl describing objects tracking states and their configuration.

    The tracker is an object used by Replica to store some states or energies.
    Thanks to extracting the state/energies storage to another object,
    we don't need to alter the Replica class to implement different storage
    strategies.

    :param initial_state: initial_state this tracker starts tracking from.
    :param initial_energy: the corresponding energy.
    """

    def __init__(self, initial_state: np.ndarray, initial_energy: float):  # pragma: no cover
        raise NotImplementedError()

    def records(self) -> Records:  # pragma: no cover
        """Return configurations stored by this tracker as a pair of sequences.

        :returns: A pair (states, energies) of sequences. The states sequence comprises
         sstates recorded by this tracker, and the energies sequence comprises
         corresponding energies.
        """
        raise NotImplementedError()

    def digest(self, new_state: np.ndarray, new_energy: float):  # pragma: no cover
        """Digest new configuration, storing or discarding it at tracker's discretion.

        Whether the state will be stored or not depends on the tracker and/or
        its configuration. For instance, trackers recording only a ground state
        approximation will only store the new state if it has lower energy
        then the previously stored one.

        :param new_state: new state for tracker to consider.
        :param new_energy: the corresponding energy.
        """
        raise NotImplementedError()


class TrackerFactory(Protocol):
    """Protocol describing a callable construting a new tracker."""
    def __call__(
        self, initial_state: np.ndarray, initial_energy: np.ndarray
    ) -> Tracker:  # pragma: no cover
        raise NotImplementedError()


class _GroundOnlyTracker:
    """Implementation of a tracker keepingtrak of a single state with lower energy.

    This tracker's `digest` command updates tracker's configuration if and only if
    it is better than the one previously stored.

    :param initial_staste: the initial_configuration of this tracker.
    :param initial_energy: the corresponding energy.
    """
    def __init__(self, initial_state, initial_energy):
        self.best_state_so_far = initial_state.copy()
        self.best_energy_so_far = initial_energy

    def records(self) -> Records:
        """Get best configuration(s) stored by this tracker.

        This implementation conforms to the Tracker protocol, and hence both elements
        of the returned tuple are lists, despite both of them having always. length 1.

        :returns: A tuple of the form ()[best_state], [best_energy])
        """
        return [self.best_state_so_far], [self.best_energy_so_far]

    def digest(self, new_state, new_energy) -> None:
        """Process new configuration.

        :param new_state: new state to process.
        :param new_energy: the corresponding energy."""
        if new_energy < self.best_energy_so_far:
            self.best_energy_so_far = new_energy
            self.best_state_so_far = new_state.copy()


@lru_cache
def _low_energy_spectrum_tracker(energy_dtype):
    @jitclass((("state", int8[:]), ("energy", energy_dtype)))
    class _HeapItem:
        def __init__(self, state, energy):
            self.state = state
            self.energy = energy

        def __lt__(self, other) -> bool:
            return self.energy < other.energy

    @jitclass((("heap", List(numba_type_of_cls(_HeapItem))), ("num_states", int64)))
    class _LowEnergySpectrumTracker:
        def __init__(self, initial_state, initial_energy, num_states):
            self.heap = [_HeapItem(initial_state.copy(), -initial_energy)]
            self.num_states = num_states

        def records(self) -> Tuple[Sequence[np.ndarray], Sequence[float]]:
            energies = np.array([-item.energy for item in self.heap])
            sorted_indices = np.argsort(energies)
            states = [self.heap[i].state for i in sorted_indices]
            return states, [energies[i] for i in sorted_indices]

        def digest(self, new_state, new_energy):
            if self._is_already_stored(new_state):
                return
            new_item = _HeapItem(new_state.copy(), -new_energy)
            heappush(self.heap, new_item)
            if len(self.heap) > self.num_states:
                heappop(self.heap)

        def _is_already_stored(self, state):
            for item in self.heap:
                if np.array_equal(state, item.state):
                    return True
            return False

    return _LowEnergySpectrumTracker


@lru_cache
def tracker_factory(energy_dtype, num_states) -> TrackerFactory:
    if num_states == 1:
        return jitclass((("best_state_so_far", int8[:]), ("best_energy_so_far", energy_dtype)))(
            _GroundOnlyTracker
        )
    else:

        def _tracker_factory(initial_state: np.ndarray, initial_energy: float) -> Tracker:
            return _low_energy_spectrum_tracker(energy_dtype)(
                initial_state, initial_energy, num_states
            )

        return _tracker_factory
