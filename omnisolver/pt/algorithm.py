"""Implementation of Parallel Tempering algorithm.

The following choice of data structures has been made to make the implementation
compatible with numba JIT (N denotes number of spins, M denotes number of replicas):

- The Ising Model being solved is represented as bias vector h_vec and
  couplings matrix j_mat.
- Connectivity information is stored as adjacency list. Adjacency list consists of:
  - a numpy array `neighbours_count` of shape (N,) declaring number of neighbours
    of each spin
  - a numpy array `adjacency_list` of shape (N, max(neighbours_count) such that
    adjacency_list[i, j] is index of j-th neighbour of i-th spin.
    If j >= neighbours_count[i], the adjacency_list[i, j] is undefined.
- State of replicas is stored in (M, N) shaped array.
- Energies of replicas are stored in (M,) shaped array.

The choice of storage format for replicas allows for an efficient parallel access
and makes it suitable to be used with numba.prange.
"""
