using ITensors, ITensorMPS
using HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using MPSTimeEvolution

include("../shared_functions.jl")

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# In this script, we "quench" the initial state (which can be loaded from a file) by
# applying a jump operator to the system site, then we proceed with the usual evolution.
# An impurity is interacting with a fermionic thermal bath, which is mapped onto two
# discrete chains by means of a thermofield+TEDOPA transformation.
# The chains are then interleaved, so that we end up with one chain only (here we are
# still dealing with the spinless case).

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_length = 1
    eps = parameters["sys_en"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["chain_coefficients"], ',', Float64; skipstart=1
    )
    emptycoups = thermofield_coefficients[:, 1]
    emptyfreqs = thermofield_coefficients[:, 3]
    filledcoups = thermofield_coefficients[:, 2]
    filledfreqs = thermofield_coefficients[:, 4]

    chain_length = parameters["chain_length"]
    total_size = 2 * chain_length + 1
    system_site = 1
    filledchain_sites = 3:2:total_size
    emptychain_sites = 2:2:total_size

    initstate_file = get(parameters, "initial_state_file", nothing)
    if isnothing(initstate_file)
        sites = siteinds("Fermion", total_size)
        initialsites = Dict(
            [
                system_site => parameters["sys_ini"]
                [st => "Occ" for st in filledchain_sites]
                [st => "Emp" for st in emptychain_sites]
            ],
        )
        ψ = MPS(sites, [initialsites[i] for i in 1:total_size])
        start_from_file = false
    else
        ψ = h5open(initstate_file, "r") do file
            return read(file, parameters["initial_state_label"], MPS)
        end
        sites = siteinds(ψ)
        start_from_file = true
        # We need to extract the site indices from ψ or else, if we define them from
        # scratch, they will have different IDs and they won't contract correctly.
    end

    @info "Apply creation operator on system site."
    q = op("c†", sites, system_site)
    ψ[system_site] = noprime(q * ψ[system_site])

    @info "Building evolution operator MPO."
    h = OpSum()
    h += eps, "n", system_site

    h +=
        emptycoups[1] *
        exchange_interaction(sites[system_site], sites[emptychain_sites[1]]) +
        filledcoups[1] *
        exchange_interaction(sites[system_site], sites[filledchain_sites[1]])

    h +=
        spin_chain(
            emptyfreqs[1:chain_length], emptycoups[2:chain_length], sites[emptychain_sites]
        ) + spin_chain(
            filledfreqs[1:chain_length],
            filledcoups[2:chain_length],
            sites[filledchain_sites],
        )

    H = MPO(h, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    operators = LocalOperator[]
    for (k, v) in parameters["observables"]
        for n in v
            push!(operators, LocalOperator(Dict(n => k)))
        end
    end
    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        if !start_from_file
            growMPS!(ψ, parameters["max_bond"])
        end
        tdvp1!(
            ψ,
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
    else
        @info "Using adaptive algorithm."
        if !start_from_file
            growMPS!(ψ, 2)
        end
        adaptivetdvp1!(
            ψ,
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
            convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
            max_bond=parameters["max_bond"],
        )
    end

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", ψ)
        end
    end
end
