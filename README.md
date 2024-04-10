# Fermionic Markovian closure

## Description of the repository
In this repository you will find, amongst other things:

* some Julia scripts which perform the simulation of some physical models such as the SIAM (`siam`) or a quantum dot impurity; (`qdot`), either using a standard TEDOPA method (`pure….jl`) or the Markovian closure (`mc….jl`);
* a `test` folder which contains the parameter files of some concrete simulations, as well as some frequently used spectral densities.
* a `mc_standard_parameters` which contains the parameters needed for the implementation of the Markovian closure technique in several simulation scripts;
* some Julia scripts that calculate the chain coefficients, in several ways, starting from the data of a spectral density (`chainmapping_….jl`);
* a folder `TimeEvoVecMPS`, a self-contained Julia package (imported by every script) which defines functions for the various flavours of the TDVP algorithm and some other utilities for the simulations.
Please note that the [`LindbladVectorizedTensors` Julia package](https://github.com/phaerrax/LindbladVectorizedTensors.jl) is required to run the simulation scripts. Please install it if it does not get installed automatically.

Each simulation script accepts a JSON file listing the parameters of the physical system. In order to run a simulation script, run *from the base folder*
```bash
julia --project=markovian_closure <script.jl> <parameter_file.json>
```
using the full relative paths of the files, e.g.
```bash
julia --project=markovian_closure siam/spinless/mc.jl test/siam/spinless/mu1/NE8/mc60_NC6.json
```

## Example
Here is an example of a complete workflow, starting from scratch.
We want to simulate a spinless SIAM with some given parameter files: `test/spectral_densities_semicircle_T4_mu0.5.json` representing a semicircle spectral density, and `examples/siam_mc.jl` with the parameters for the physical simulation of the model with a Markovian closure.
1. Generate the thermofield coefficients from the spectral density, with
```bash
julia --project=markovian_closure chainmapping_thermofield.jl test/spectral_densities/semicircle_T4_mu0.5.json
```
The outputs a file called `test/spectral_densities/semicircle_T4_mu0.5.thermofield` which will be called later in `examples/siam_mc.json` as the `thermofield_coefficients` entry.
It it not necessary to generate the coefficient each time, if they already exist.
2. Run the simulation script with
```bash
julia --project=markovian_closure siam/spinless/mc.jl examples/siam_mc.json
```
The observables given in the parameter file must be existing `ITensors` operator names. 
Note that the parameter file `examples/siam_mc.json` also specifies some output files which will contain the expectation values of the given observables, an HDF5 file containing the final state, and so on. If one does not need such results, `/dev/null` or an equivalent destination may be given to avoid creating unnecessary output files.

## Documentation
The `TimeEvoVecMPS` package has its own documentation (still a work in progress), which is
not yes hosted and must be generated manually.
From the `docs` directory, run `julia --project make.jl` and wait for the command to end.
You will be able to access the documentation from `docs/builds/index.html`, by opening it
from any Web browser.
