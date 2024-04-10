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
    emptycoups = thermofield_coefficients[:, 1]
    emptyfreqs = thermofield_coefficients[:, 3]
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    # -------------------------
    empty_Ω = parameters["empty_asympt_frequency"]
    empty_K = parameters["empty_asympt_coupling"]

    α = readdlm(parameters["MC_alphas"])
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    emptymc = closure(empty_Ω, empty_K, α, β, w)
    closure_length = length(emptymc)

    # Site ranges
    system_site = 1
    empty_chain_range = range(; start=2, step=1, length=chain_length)
    empty_closure_range = range(;
        start=empty_chain_range[end] + 1, step=1, length=closure_length
    )

    total_size = 1 + chain_length + closure_length

    sites = siteinds("Fermion", total_size)
    initialsites = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "Emp" for st in empty_chain_range]
            [st => "Emp" for st in empty_closure_range]
        ],
    )
    ψ = MPS(sites, [initialsites[i] for i in 1:total_size])

    h_chain = spin_chain(
        [ε; emptyfreqs[1:chain_length]],
        emptycoups[1:chain_length],
        sites[1:empty_chain_range[end]],
    )

    h_effclosure = spin_chain(
        freqs(emptymc), innercoups(emptymc), sites[empty_closure_range]
    )
    for (i, site) in enumerate(empty_closure_range)
        h_effclosure += outercoup(emptymc, i), "c†", empty_chain_range[end], "c", site
        h_effclosure += conj(outercoup(emptymc, i)), "c†", site, "c", empty_chain_range[end]
        h_effclosure += -0.5im * damp(emptymc, i), "n", site
    end

    H = MPO(h_chain + h_effclosure, sites)

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
