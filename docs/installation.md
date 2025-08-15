Installation
===============

This page details how to get started with `harbor` and how to install it on your system.

There are two ways to install `asapdiscovery`:

1. From conda-forge (recommended)
2. Developer installation from source

Installation from conda-forge
----------------------------
```{note}
This is underway!
```

Developer installation from source
----------------------------------

To install `harbor` from source, you will need to clone the repository, setup a compatible environment with mamba and install the package using `pip`. You can do this using the following commands:

```bash
git clone git@github.com:choderalab/harbor.git
cd harbor
mamba env create -f devtools/harbor.yml # chose relevant file for your platform
mamba activate harbor
```
