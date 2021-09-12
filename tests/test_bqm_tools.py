"""Test cases for utility functions used in PT sampler."""
import dimod
import numpy as np

from omnisolver.pt.bqm_tools import adjacency_list_from_couplings, vectorize_bqm


class TestBQMVectorization:
    def test_dense_bqm_vectorization_comprises_correct_biases_and_couplings(self):
        bqm = dimod.BQM(
            {0: -1, 1: 0.5, 2: -3.0},
            {(0, 1): -2.5, (0, 2): 0.3, (1, 2): 0.1},
            0,
            vartype="SPIN",
        )

        expected_h_vec = np.array([-1.0, 0.5, -3.0])
        expected_j_mat = np.array([[0.0, -2.5, 0.3], [-2.5, 0.0, 0.1], [0.3, 0.1, 0.0]])

        h_vec, j_mat = vectorize_bqm(bqm)

        np.testing.assert_equal(h_vec, expected_h_vec)
        np.testing.assert_equal(j_mat, expected_j_mat)

    def test_sparse_bqm_vectorization_contains_zeros_for_missing_coefficients(self):
        bqm = dimod.BQM({0: -10, 3: 0.5}, {(1, 2): -2.5}, 0, vartype="SPIN")

        expected_h_vec = np.array([-10.0, 0.0, 0.0, 0.5])
        expected_j_mat = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -2.5, 0.0],
                [0.0, -2.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        h_vec, j_mat = vectorize_bqm(bqm)

        np.testing.assert_equal(h_vec, expected_h_vec)
        np.testing.assert_equal(j_mat, expected_j_mat)


class TestAdjacencyListConstruction:
    def test_neighbours_of_all_spins_in_dense_graph_are_correctly_identified(self):
        j_mat = np.array(
            [
                [0.0, 3.0, -1.0, 2.0],
                [3.0, 0.0, -0.5, -1.2],
                [-1.0, -0.5, 0.0, -0.7],
                [2.0, -1.2, -0.7, 0],
            ]
        )

        expected_adjacency_list = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

        expected_neighbours_count = 3 * np.ones(4)

        adjacency_list, neighbours_count = adjacency_list_from_couplings(j_mat)

        np.testing.assert_equal(neighbours_count, expected_neighbours_count)
        np.testing.assert_equal(adjacency_list, expected_adjacency_list)

    def test_neighbours_of_all_spins_in_sparse_graph_are_correctly_identified(self):
        j_mat = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -2.0, 0.0, 0.0],
                [1.0, -2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.5, 0.0],
            ]
        )

        expected_neighbours_count = np.array([1, 1, 2, 1, 1])

        expected_adjacency_list = np.array([[2, 0], [2, 0], [0, 1], [4, 0], [3, 0]])

        adjacency_list, neighbours_count = adjacency_list_from_couplings(j_mat)

        np.testing.assert_equal(neighbours_count, expected_neighbours_count)
        np.testing.assert_equal(adjacency_list, expected_adjacency_list)
