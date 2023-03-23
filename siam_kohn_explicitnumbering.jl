using ITensors
using IterTools
using DelimitedFiles
using PseudomodesTTEDOPA

include("./TDVP_lib_VecRho.jl")

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a fermionic thermal bath, which is mapped onto two
# discrete chains by means of a thermofield+TEDOPA transformation.
#
# Contrary to `siam_kohn.jl`, in this script an explicit site numbering for the chains
# is used. This serves the purpose of checking whether the more intuitive enumeration
# of the sites used in the “twin” script is correct (it is).

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    rcoups = thermofield_coefficients[:, 1]
    rfreqs = thermofield_coefficients[:, 3]
    lcoups = thermofield_coefficients[:, 2]
    lfreqs = thermofield_coefficients[:, 4]

    NE = parameters["chain_length"]
    total_size = 2NE + 1

    sites = siteinds("Fermion", total_size)
    psi0 = MPS(
        sites,
        [repeat(["Occ"], NE); system_initstate; repeat(["Emp"], NE)],
    )

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, parameters["max_bond"])

    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += eps, "n", NE+1

    # - system-chain interaction
    h += lcoups[1], "c†", NE+1, "c", NE
    h += lcoups[1], "c†", NE, "c", NE+1

    h += rcoups[1], "c†", NE+1, "c", NE+2
    h += rcoups[1], "c†", NE+2, "c", NE+1

    # - chain terms
    for j in 1:NE
        h += lfreqs[j], "n", NE-j+1
        h += rfreqs[j], "n", NE+j+1
    end
    for j in 1:(NE-1)
        h += lcoups[j + 1], "c†", NE-j+1, "c", NE-j
        h += lcoups[j + 1], "c†", NE-j, "c", NE-j+1
        h += rcoups[j + 1], "c†", NE+j+1, "c", NE+j+2
        h += rcoups[j + 1], "c†", NE+j+2, "c", NE+j+1
    end

    H = MPO(h, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    obs = []
    oblist = parameters["observables"]
    for key in keys(oblist)
        foreach(i -> push!(obs, [key, i]), oblist[key])
    end

    cb = LocalPosMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )

    tdvp1!(
        psi,
        H,
        timestep,
        tmax;
        hermitian=true,
        normalize=false,
        callback=cb,
        progress=true,
        exp_tol=parameters["exp_tol"],
        krylovdim=parameters["krylov_dim"],
        store_psi0=false,
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
    )
end
