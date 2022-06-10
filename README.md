# epiphy

Epistatic selection models in phylogenetics.

## Installation

Recommended prerequisites are [miniconda](https://docs.conda.io/en/latest/miniconda.html) with [mamba](https://github.com/mamba-org/mamba) installed in the base environment. Choose an appropriate architecture `$ARCH` (both Intel and Apple processors are supported, as well as Linux) and run:

```
mamba env create -f environment-$ARCH.yml
```

## Usage

More to come, but for now activate the above environment and:

```
snakemake -j 1
```