using ITensors
using ITensors.HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using TimeEvoVecMPS

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    ε = parameters["sys_en"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    filledcoups = thermofield_coefficients[:, 2]
    filledfreqs = thermofield_coefficients[:, 4]
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    # -------------------------
    filled_Ω = parameters["filled_asympt_frequency"]
    filled_K = parameters["filled_asympt_coupling"]

    α = readdlm(parameters["MC_alphas"])
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    filledmc = closure(filled_Ω, filled_K, α, β, w)
    closure_length = length(filledmc)

    # Site ranges
    system_site = 1
    filled_chain_range = range(; start=2, step=1, length=chain_length)
    filled_closure_range = range(;
        start=filled_chain_range[end] + 1, step=1, length=closure_length
    )

    total_size = 1 + chain_length + closure_length

    sites = siteinds("Fermion", total_size)
    initialsites = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "Occ" for st in filled_chain_range]
            [st => "Occ" for st in filled_closure_range]
        ],
    )
    ψ = MPS(sites, [initialsites[i] for i in 1:total_size])

    h_chain = spin_chain(
        [ε; filledfreqs[1:chain_length]],
        filledcoups[1:chain_length],
        sites[1:filled_chain_range[end]],
    )

    h_effclosure = spin_chain(
        freqs(filledmc), innercoups(filledmc), sites[filled_closure_range]
    )
    for (i, site) in enumerate(filled_closure_range)
        h_effclosure += outercoup(filledmc, i), "c†", filled_chain_range[end], "c", site
        h_effclosure += conj(outercoup(filledmc, i)),
        "c†", site, "c",
        filled_chain_range[end]
        h_effclosure += -0.5im * damp(filledmc, i), "Id", site
        h_effclosure += 0.5im * damp(filledmc, i), "n", site
    end

    H = MPO(h_chain + h_effclosure, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    operators = LocalOperator[]
    for (k, v) in parameters["observables"]
        for n in v
            push!(operators, LocalOperator(Dict(n => k)))
        end
    end
    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

    growMPS!(ψ, parameters["max_bond"])

    tdvp1!(
        ψ,
        H,
        timestep,
        tmax;
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        ishermitian=false,
        issymmetric=false,
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
    )

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", ψ)
        end
    end
end
