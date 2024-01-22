# Fermionic Markovian closure
In this repository you will find, amongst other things:

* some Julia scripts which perform the simulation of some physical models such as the SIAM (`siam`) or a quantum dot impurity; (`qdot`), either using a standard TEDOPA method (`pure….jl`) or the Markovian closure (`mc….jl`);
* a `test` folder which contains the parameter files of some concrete simulations, as well as some frequently used spectral densities.
* a `mc_standard_parameters` which contains the parameters needed for the implementation of the Markovian closure technique in several simulation scripts;
* some Julia scripts that calculate the chain coefficients, in several ways, starting from the data of a spectral density (`chainmapping_….jl`);
* a folder `TimeEvoVecMPS`, a self-contained Julia package (imported by every script) which defines functions for the various flavours of the TDVP algorithm and some other utilities for the simulations.
Please note that the [`PseudomodesTTEDOPA` Julia package](https://github.com/phaerrax/PseudomodesTTEDOPA.jl) is required to run the simulation scripts. Please install it if it does not get installed automatically.

Each simulation script accepts a JSON file listing the parameters of the physical system. In order to run a simulation script, run *from the base folder*
```bash
julia --project=markovian_closure <script.jl> <parameter_file.jl>
```
using the full relative paths of the files, e.g.
```bash
julia --project=markovian_closure siam/spinless/mc.jl test/siam/spinless/mu1/NE8/mc60_NC6.json
```
