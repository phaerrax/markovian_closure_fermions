using ITensors
using ITensors.HDF5
using DelimitedFiles
using PseudomodesTTEDOPA
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
    total_size = chain_length + 1
    systempos = 1
    emptychain_sites = 2:total_size

    sites = siteinds("Fermion", total_size)
    initialsites = Dict(
        [
            systempos => parameters["sys_ini"]
            [st => "Emp" for st in emptychain_sites]
        ]
    )
    psi = MPS(sites, [initialsites[i] for i in 1:total_size])

    H = MPO(
        spin_chain([ε; emptyfreqs[1:chain_length]], emptycoups[1:chain_length], sites),
        sites,
    )

    obs = []
    oblist = parameters["observables"]
    for key in keys(oblist)
        foreach(i -> push!(obs, [key, i]), oblist[key])
    end

    cb = LocalPosMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * parameters["tstep"]
    )

    growMPS!(psi, parameters["max_bond"])

    tdvp1!(
        psi,
        H,
        parameters["tstep"],
        parameters["tmax"];
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        ishermitian=true,
        issymmetric=false,
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
    )

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", psi)
        end
    end
end
