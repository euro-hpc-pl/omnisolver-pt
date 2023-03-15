
![Logo](https://raw.githubusercontent.com/euro-hpc-pl/omnisolver/master/logo.png)
*Parallel Tempering Plugin for [Omnisolver](https://github.com/euro-hpc-pl/omnisolver)*


## Installation 

To install the plugin run:

```shell
pip install omnisolver-pt
```

## Command line usage

```text
usage: omnisolver pt [-h] [--output OUTPUT] [--vartype {SPIN,BINARY}] [--num_replicas NUM_REPLICAS] [--num_pt_steps NUM_PT_STEPS]
                     [--num_sweeps NUM_SWEEPS] [--beta_min BETA_MIN] [--beta_max BETA_MAX]
                     input

Parallel tempering sampler

positional arguments:
  input                 Path of the input BQM file in COO format. If not specified, stdin is used.

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Path of the output file. If not specified, stdout is used.
  --vartype {SPIN,BINARY}
                        Variable type
  --num_replicas NUM_REPLICAS
                        number of replicas to simulate (default 10)
  --num_pt_steps NUM_PT_STEPS
                        number of parallel tempering steps
  --num_sweeps NUM_SWEEPS
                        number of Monte Carlo sweeps per parallel tempering step
  --beta_min BETA_MIN   inverse temperature of the hottest replica
  --beta_max BETA_MAX   inverse temperature of the coldest replica
```

## Citing

If you used the Omnisolver package or one of its plugins, please cite:

```text
@misc{https://doi.org/10.48550/arxiv.2112.11131,
  doi = {10.48550/ARXIV.2112.11131},
  
  url = {https://arxiv.org/abs/2112.11131},
  
  author = {Jałowiecki, Konrad and Pawela, Łukasz},
  
  keywords = {Software Engineering (cs.SE), Quantum Physics (quant-ph), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences},
  
  title = {Omnisolver: an extensible interface to Ising spin glass solvers},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
