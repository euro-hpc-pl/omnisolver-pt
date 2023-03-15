# Welcome to omnisolver-pt documentation!

The `omnisolver-pt` is Omnisolver plugin implementing the Parallel Tempering algorithm.

## Installation

Preferred method of installation is via pip:

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

## Documentation
```{toctree}
:maxdepth: 1
```
