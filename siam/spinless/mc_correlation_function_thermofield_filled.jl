using ITensors, ITensorMPS
using HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using MarkovianClosure
using MPSTimeEvolution

include("../shared_functions.jl")

let
    parameters = load_pars(ARGS[1])

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

    total_size = 2chain_length + 2closure_length

    # Site ranges
    # We switch the positions of the empty and filled branches, so that the filled one
    # starts at 1. This way we don't have to deal with Jordan-Wigner strings.
    empty_chain_range = range(; start=2, step=2, length=chain_length)
    empty_closure_range = range(;
        start=empty_chain_range[end] + 2, step=2, length=closure_length
    )
    filled_chain_range = range(; start=1, step=2, length=chain_length)
    filled_closure_range = range(;
        start=filled_chain_range[end] + 2, step=2, length=closure_length
    )
    @assert empty_closure_range[end] == total_size

    # We are going to compute ⟨ c†(t) c(0) ⟩ on the free environment.
    # This is the only relevant correlation function of the filled branch of the environment
    # that determines the reduced dynamics. We can leave out the system in this simulation.
    # Since ⟨vec(A), vec(B)⟩ = tr(A† B), and c, c† aren't obviously self-adjoint, we must
    # be careful:
    # ⟨ c†(t) c(0) ⟩ = tr( c ρ₀ c†(t) ) = tr( (ρ₀ c†)† c†(t) ) = ⟨vec(ρ₀ c†), vec(c†(t))⟩.
    sites = siteinds("vFermion", total_size)
    initialsites = Dict(
        [
            [st => "Up" for st in filled_chain_range]
            [st => "Up" for st in filled_closure_range]
            [st => "Dn" for st in empty_chain_range]
            [st => "Dn" for st in empty_closure_range]
        ],
    )
    ρ₀cdag₁ =
        filled_chain_coups[1] *
        MPS(ComplexF64, sites, [initialsites[i] for i in 1:total_size])
    ρ₀cdag₁[filled_chain_range[1]] = noprime(
        op("⋅σ+", sites, filled_chain_range[1]) * ρ₀cdag₁[filled_chain_range[1]]
    )
    start_from_file = false

    initialops = Dict(
        [
            [st => "vCdag" for st in filled_chain_range[1]]
            [st => "vId" for st in filled_chain_range[2:end]]
            [st => "vId" for st in filled_closure_range]
            [st => "vId" for st in empty_chain_range]
            [st => "vId" for st in empty_closure_range]
        ],
    )
    cdag₁ =
        filled_chain_coups[1] *
        MPS(ComplexF64, sites, [initialops[i] for i in 1:total_size])
    opgrade = -1 # (odd parity)

    L′ = MPO(
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

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        if !start_from_file
            growMPS!(cdag₁, parameters["max_bond"])
        end
        adjtdvp1vec!(
            cdag₁,
            ρ₀cdag₁,
            L′,
            parameters["tstep"],
            parameters["tmax"],
            parameters["ms_stride"] * parameters["tstep"];
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
        )
    else
        @info "Using adaptive algorithm."
        if !start_from_file
            growMPS!(cdag₁, 4)
        end
        adaptiveadjtdvp1vec!(
            cdag₁,
            ρ₀cdag₁,
            L′,
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
        )
    end
end
