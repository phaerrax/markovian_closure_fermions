using ITensors
using ITensors.HDF5
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a fermionic thermal bath, which is mapped onto two
# discrete chains by means of a thermofield+TEDOPA transformation.
# The Markovian closure technique is then applied to _both_ chains, truncating the two
# environments and replacing part of them with sets of pseudomodes.
# The two chains are then interleaved so as to build a single chain. This is hardcoded,
# for now; maybe sometime I'll find a way to calculate the Jordan-Wigner operators in
# an automatic way...

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    # ------------------------
    system_length = 1
    eps = parameters["sys_en"]

    # Input: chain stub parameters
    # ----------------------------
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    empty_chain_coups = thermofield_coefficients[:, 1]
    empty_chain_freqs = thermofield_coefficients[:, 3]
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    # -------------------------
    empty_Ω = parameters["empty_asympt_frequency"]
    empty_K = parameters["empty_asympt_coupling"]

    α_mat = readdlm(parameters["MC_alphas"])
    β_mat = readdlm(parameters["MC_betas"])
    w_mat = readdlm(parameters["MC_coups"])

    α = α_mat[:, 1] .+ im .* α_mat[:, 2]
    β = β_mat[:, 1] .+ im .* β_mat[:, 2]
    w = w_mat[:, 1] .+ im .* w_mat[:, 2]

    emptymc = closure(empty_Ω, empty_K, α, β, w)
    closure_length = length(emptymc)

    total_size = system_length + chain_length + closure_length

    # Site ranges
    system_site = 1
    empty_chain_range = range(; start=2, step=1, length=chain_length)
    empty_closure_range = range(;
        start=empty_chain_range[end] + 1, step=1, length=closure_length
    )
    @assert empty_closure_range[end] == total_size

    initstate_file = get(parameters, "initial_state_file", nothing)
    if isnothing(initstate_file)
    system_initstate = parameters["sys_ini"]
        sites = siteinds("vFermion", total_size)
        initialsites = Dict(
            [
                system_site => parameters["sys_ini"]
                [st => "Dn" for st in empty_chain_range]
                [st => "Dn" for st in empty_closure_range]
            ],
        )
        vecρₜ = MPS(sites, [initialsites[i] for i in 1:total_size])
        start_from_file = false
    else
        vecρₜ = h5open(initstate_file, "r") do file
            return read(file, parameters["initial_state_label"], MPS)
        end
        sites = siteinds(vecρₜ)
        start_from_file = true
        # We need to extract the site indices from vecρₜ or else, if we define them from
        # scratch, they will have different IDs and they won't contract correctly.
    end

    L = MPO(
        eps * gkslcommutator("N", system_site) +
        empty_chain_coups[1] *
        exchange_interaction(sites[system_site], sites[empty_chain_range[1]]) +
        spin_chain(
            empty_chain_freqs[1:chain_length],
            empty_chain_coups[2:chain_length],
            sites[empty_chain_range],
        ) +
        closure_op(emptymc, sites[empty_closure_range], empty_chain_range[end]),
        sites,
    )

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    obs = []
    oblist = parameters["observables"]
    for key in keys(oblist)
        foreach(i -> push!(obs, [key, i]), oblist[key])
    end

    cb = LocalPosVecMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        if !start_from_file
            growMPS!(vecρₜ, parameters["max_bond"])
        end
        tdvp1vec!(
            vecρₜ,
            L,
            timestep,
            tmax,
            sites;
            hermitian=false,
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
            growMPS!(vecρₜ, 2)
        end
        adaptivetdvp1vec!(
            vecρₜ,
            L,
            timestep,
            tmax,
            sites;
            hermitian=false,
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
            write(f, "final_state", vecρₜ)
        end
    end
end
