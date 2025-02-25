using ITensors, ITensorMPS
using HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using MarkovianClosure
using MPSTimeEvolution

include("../shared_functions.jl")

let
    parameters = load_pars(ARGS[1])

    # Load initial state from disk
    initstate_file = get(parameters, "initial_state_file", nothing)
    vecstate_0 = h5open(initstate_file, "r") do file
        return read(file, parameters["initial_state_label"], MPS)
    end
    sites = siteinds(vecstate_0)

    # Rescale the initial state so that its trace is 1. It should already be 1, ideally,
    # but if it comes from a previous TDVP evolution it might have deviated a little from
    # this value. As long as it is _a little_ different from 1, rescaling is fine.
    trvecstate_0 = real(dot(MPS(sites, "vId"), vecstate_0))
    vecstate_0 /= trvecstate_0

    # Input: system parameters
    # ------------------------
    system_length = 1
    ε = parameters["sys_en"]

    # Input: chain stub parameters
    # ----------------------------
    thermofield_coefficients = readdlm(
        parameters["chain_coefficients"], ',', Float64; skipstart=1
    )
    empty_chain_coups = thermofield_coefficients[:, 1]
    empty_chain_freqs = thermofield_coefficients[:, 3]
    filled_chain_coups = thermofield_coefficients[:, 2]
    filled_chain_freqs = thermofield_coefficients[:, 4]
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    # -------------------------
    empty_Ω = meanordefault(
        empty_chain_freqs[(chain_length + 1):end],
        get(parameters, "empty_asympt_frequency", nothing),
    )
    empty_K = meanordefault(
        empty_chain_coups[(chain_length + 1):end],
        get(parameters, "empty_asympt_coupling", nothing),
    )
    filled_Ω = meanordefault(
        filled_chain_freqs[(chain_length + 1):end],
        get(parameters, "filled_asympt_frequency", nothing),
    )
    filled_K = meanordefault(
        filled_chain_coups[(chain_length + 1):end],
        get(parameters, "filled_asympt_coupling", nothing),
    )

    α_mat = readdlm(parameters["MC_alphas"])
    β_mat = readdlm(parameters["MC_betas"])
    w_mat = readdlm(parameters["MC_coups"])

    α = α_mat[:, 1] .+ im .* α_mat[:, 2]
    β = β_mat[:, 1] .+ im .* β_mat[:, 2]
    w = w_mat[:, 1] .+ im .* w_mat[:, 2]

    emptymc = markovianclosure_parameters(empty_Ω, empty_K, α, β, w)
    filledmc = markovianclosure_parameters(filled_Ω, filled_K, conj.(α), conj.(β), w)
    closure_length = length(emptymc)

    # Site ranges
    system_site = 1
    empty_chain_range = range(; start=2, step=2, length=chain_length)
    empty_closure_range = range(;
        start=empty_chain_range[end] + 2, step=2, length=closure_length
    )
    filled_chain_range = range(; start=3, step=2, length=chain_length)
    filled_closure_range = range(;
        start=filled_chain_range[end] + 2, step=2, length=closure_length
    )

    total_size = system_length + 2chain_length + 2closure_length
    total_size != length(vecstate_0) && error(
        "Number of sites between initial state from file and parameters does not match."
    )

    # We compute here the retarded Green's function Gr(t) = -i tr({d(t), d*(0)} ρ). We can
    # write it as
    #   Gr(t) = -i tr(d(t) d* ρ) -i tr(d* d(t) ρ) = -i tr(d(t) d* ρ) -i tr(d(t) ρ d*)
    # so we can evolve d(t) once and measure simultaneaously its expectation values on
    # both d* ρ and ρ d*.
    # [Note that we actually need to evolve d* since the vectorization is computed using the
    # Hilbert-Schmidt inner product (x, y) = tr(x* y).]
    # TODO How about directly using d* ρ + ρ d* as initial state? 

    # Creation/annihilation operators aren't Hermitian so we need a complex vector to
    # represent them.
    opstrings = Dict(
        [
            system_site => "vAdag"
            [st => "vId" for st in filled_chain_range]
            [st => "vId" for st in filled_closure_range]
            [st => "vId" for st in empty_chain_range]
            [st => "vId" for st in empty_closure_range]
        ],
    )
    targetop = MPS(ComplexF64, sites, opstrings)
    growMPS!(targetop, parameters["max_bond"])
    opgrade = -1  # (odd parity)

    adjL = MPO(
        -ε * gkslcommutator("N", system_site) +
        empty_chain_coups[1] *
        exchange_interaction′(sites[system_site], sites[empty_chain_range[1]]) +
        filled_chain_coups[1] *
        exchange_interaction′(sites[system_site], sites[filled_chain_range[1]]) +
        spin_chain′(
            empty_chain_freqs[1:chain_length],
            empty_chain_coups[2:chain_length],
            sites[empty_chain_range],
        ) +
        spin_chain′(
            filled_chain_freqs[1:chain_length],
            filled_chain_coups[2:chain_length],
            sites[filled_chain_range],
        ) +
        markovianclosure′(
            emptymc, sites[empty_closure_range], empty_chain_range[end], opgrade
        ) +
        filled_markovianclosure′(
            filledmc, sites[filled_closure_range], filled_chain_range[end], opgrade
        ),
        sites,
    )

    # The Green's function can be retrieved by summing the columns exp_val_a*rho_im
    # and exp_val_rho*a_im
    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        adjtdvp1vec!(
            targetop,
            # Apply creation operator to the initial state, on the left and on the right
            [
                apply(op("A†⋅", sites, system_site), vecstate_0),
                apply(op("⋅A†", sites, system_site), vecstate_0),
            ],
            adjL,
            parameters["tstep"],
            parameters["tmax"],
            parameters["ms_stride"] * parameters["tstep"];
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
            initialstatelabels=["a*rho", "rho*a"],
        )
    else
        @info "Using adaptive algorithm."
        adaptiveadjtdvp1vec!(
            targetop,
            [
                apply(op("A†⋅", sites, system_site), vecstate_0),
                apply(op("⋅A†", sites, system_site), vecstate_0),
            ],
            adjL,
            parameters["tstep"],
            parameters["tmax"],
            parameters["ms_stride"] * parameters["tstep"];
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
            convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
            max_bond=parameters["max_bond"],
            initialstatelabels=["a*rho", "rho*a"],
        )
    end
end
