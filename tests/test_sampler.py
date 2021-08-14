"""Test cases for PTSampler."""
import dimod

from omnisolver.pt.sampler import PTSampler


class TestSamplingIsingModel:
    def test_resulting_state_and_energy_agree(self):
        biases = {0: 0.5, 1: -2.0, 2: 3.0}
        couplings = {(0, 1): -1, (2, 1): -3}
        sampler = PTSampler()

        result = sampler.sample_ising(
            biases,
            couplings,
            num_replicas=3,
            num_pt_steps=100,
            num_sweeps=10,
            beta_min=0.01,
            beta_max=0.1,
        )

        bqm = dimod.BQM.from_ising(biases, couplings)
        assert bqm.energy(result.first.sample) == result.first.energy
