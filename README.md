# Fermionic Markovian closure

This repository contains the code used for the simulations in [[1]](#1).
There are some scripts computing the evolution of several types of physical
systems, employing the Markovian closure technique as explained in the article.

## Installation

1) Clone the repository on your computer.
2) This project depends on some packages that aren't found in Julia's general
    registry, but in [this one](https://github.com/phaerrax/TensorNetworkSimulations),
    so you need to add it first in order to download those packages
    automatically (you can skip this step if you already added the registry
    previously: you need to do it just once per Julia installation). Open a
    Julia interactive session and enter the Pkg REPL by hitting `]` (see the
    [Getting Started with Environments](https://pkgdocs.julialang.org/v1/getting-started/#Getting-Started-with-Environments)
    guide) and then run `registry add https://github.com/phaerrax/TensorNetworkSimulations.git`:

    ```julia-repl
    (@v1.11) pkg> registry add https://github.com/phaerrax/TensorNetworkSimulations.git
         Cloning registry from "https://github.com/phaerrax/TensorNetworkSimulations.git"
           Added registry `TensorNetworkSimulations` to `~/.julia/registries/TensorNetworkSimulations`
    ```

    (If this is the first time using Julia then run `registry add General` too
    otherwise the `General` registry won't be automatically added.)

3) Activate the Julia project in this folder with `activate .`

    ```julia-repl
    (@v1.11) pkg> activate .
      Activating project at `~/markovian_closure_fermions`
    ```

4) Add the `OpenSystemsChainMapping` package and download all the
    dependencies by running `dev ./OpenSystemsChainMapping` and then
    `instantiate`. 

## Description of the repository

The repository contains a package, called `OpenSystemsChainMapping`, which
defines some "core" functions that allow simulating some open quantum systems
in contact with one or two environments.

You will find:

* some Julia scripts which perform the simulation of some physical models such
  as the SIAM (`siam`) or a quantum dot impurity (`qdot`), either using a
  standard TEDOPA method (`pure….jl`) or the Markovian closure (`mc….jl`);
* an `example` folder which contains the parameter files of some concrete
  simulations, as well as some frequently used spectral densities.
* some Julia scripts that calculate the chain coefficients, in several ways,
  starting from the data of a spectral density (`chainmapping_….jl`);

## How it works (WIP)

The parameters for each simulation script must be supplied, in a JSON file, as
the first (and only) command-line argument. You can find some examples in the
`test/spectral_densities` directory. Please note that the script that computes
the thermofield coefficients currently works only if the chemical potential is
zero (this does not affect the generality of our algorithms, since any spectral
density can always be shifted so that its chemical potential is zero), so for
now care must be taken to write the parameter files according to this
specification. All files in `test/spectral_densities` already follow this
convention.
In order to run a simulation script, run *from the base folder* (i.e. the root
of the Git repository)

```bash
julia --project <script.jl> <parameter_file.json>
```

using the full relative paths of the files, e.g.

```bash
julia --project siam/spinless/mc.jl test/siam/spinless/mu1/NE8/mc60_NC6.json
```

## Example

Here is an example of a complete workflow, starting from scratch.
We want to simulate a spinless SIAM with some given parameter files:
`test/spectral_densities_semicircle_T4_mu0.5.json` representing a semicircle
spectral density, and `examples/siam_mc.jl` with the parameters for the
physical simulation of the model with a Markovian closure.

1. Generate the thermofield coefficients from the spectral density, with

  ```bash
  julia --project chainmapping_thermofield.jl examples/spectral_densities/semicircle_T4_mu0.5.json
  ```

  The output is a file called
  `test/spectral_densities/semicircle_T4_mu0.5.thermofield`, which will be called
  later in `examples/siam_mc.json` in the `chain_coefficients` entry.
  It is not necessary to generate the coefficient each time, if the file already
  exists.
2. Run the simulation script with

  ```bash
  julia --project siam/spinless/mc.jl examples/siam_spinless_mc.json
  ```

  Note that the parameter file `examples/siam_mc.json` also specifies some
  output files which will contain the expectation values of the given
  observables, an HDF5 file containing the final state, and so on.
  If one does not need such results, `/dev/null` or an equivalent destination
  may be given to avoid creating unnecessary output files.

The `examples` directory contains other sample parameter files that can be
used with other simulation scripts:

```bash
julia --project siam/spinless/pure.jl examples/siam_spinless_pure.json
julia --project siam/spinful/mc.jl examples/siam_spinful_mc.json
julia --project qdot/mc_2l.jl examples/qdot_2levels_mc.json
```

## Common parameters and their meaning

* Chain coefficients:

    - `chain_coefficients`: a file containing the chain coefficients
    - `chain_length`: an integer specifying how many sites to keep in the
      environment chains before attaching the closure

* Output files:

    - `out_file`: expectation values of observables
    - `state_file`: final MPS (in HDF5 format)
    - `ranks_file`: bond dimensions of the MPS, step by step
    - `times_file`: wall-clock time needed for each evolution step

* Integration parameters for the simulation:

    - `tmax`: final (physical) time of the simulation
    - `tstep`: integration time step
    - `ms_stride`: measure observables only each `ms_stride` steps
    - `max_bond`: maximum bond dimension for the MPS
    - `discarded_w`: cutoff for MPS truncation during the evolution

* Observables:

    - `observables`: a vector specifying the name of the observable (a valid
    ITensor operator) and a list of sites on which it will be measured, e.g.

        ```json
        {
            "vN":  [1,2,3,10,11,16,17],
            "vX":  [1,2,3,4]
        }
        ```

    (note that the `v` prefix here is used for the "vectorized fermion" systems
    --- please consult the `LindbladVectorizedTensors` package documentation for
    more information)

* Markovian closure parameters:

    - `MC_alphas`: `mc_standard_parameters/alphas_6.dat`,
    - `MC_betas`: `mc_standard_parameters/betas_6.dat`,
    - `MC_coups`: `mc_standard_parameters/coupls_6.dat`

    These are already existing files. You only need to change the number in the
    file name from 6 to 8 or 10 if you want to use closures with a different
    number of modes.

* Other system-specific parameters, e.g. when the open system is a two-level
system (a fermion, maybe) we need:

    - `sys_en`: energy of the open system
    - `sys_ini`: initial state of the open system

## References

<a id="1">[1]</a>
Ferracin, D., Smirne, A., Huelga, S. F., Plenio M. B. and Tamascelli, D. (2024).
_Spectral Density Modulation and Universal Markovian Closure of Fermionic
Environments_. [arxiv.org/abs/2407.10017](https://arxiv.org/abs/2407.10017).
