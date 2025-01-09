using ITensors, ITensorMPS, MPSTimeEvolution, CSV

include("../shared_functions.jl")

function simulation(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcoupling,
    nenvironment,
    environment_chain_frequencies,
    environment_chain_couplings,
    dt,
    tmax,
    maxbonddim,
    io_file=nullfile(),
    io_ranks=nullfile(),
    io_times=nullfile(),
    operators,
)
    sites = siteinds("Fermion", 2nsystem + 2nenvironment; conserve_nfparity=true)
    system = ModeChain(range(; start=1, step=2, length=nsystem), [system_energy], [])
    altsystem = ModeChain(range(; start=2, step=2, length=nsystem), [system_energy], [])

    environment = ModeChain(
        range(; start=2nsystem + 1, step=2, length=nenvironment),
        environment_chain_frequencies,
        environment_chain_couplings,
    )
    altenvironment = ModeChain(
        range(; start=2nsystem + 2, step=2, length=nenvironment),
        environment_chain_frequencies,
        environment_chain_couplings,
    )

    initstate = MPS(sites, n -> n ≤ 2nsystem ? "Occ" : "Emp")

    ad_h =
        spinchain(SiteType("Fermion"), join(system, environment, sysenvcoupling)) -
        spinchain(SiteType("Fermion"), join(altsystem, altenvironment, sysenvcoupling))

    L = MPO(-im * ad_h, sites)

    cb = SuperfermionCallback(operators, sites, dt)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    state_t = enlargelinks(
        initstate, maxbonddim; ref_state=n -> n ≤ 2nsystem ? "Occ" : "Emp"
    )

    simulation_files_info(;
        measurements_file=io_file, bonddims_file=io_ranks, simtime_file=io_times
    )

    tdvp1vec!(
        state_t,
        L,
        dt,
        tmax,
        sites;
        hermitian=false,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=io_file,
        io_ranks=io_ranks,
        io_times=io_times,
        superfermions=true,
    )

    return nothing
end

function main()
    parsedargs = parsecommandline()

    chain_length = parsedargs["environment_sites"]

    empty_chain_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    empty_chain_coups = CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]

    ops = LocalOperator[]
    for (k, v) in parsedargs["observables"]
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end

    measurements_file = parsedargs["output"] * "_measurements.csv"
    bonddims_file = parsedargs["output"] * "_bonddims.csv"
    simtime_file = parsedargs["output"] * "_simtime.csv"

    simulation(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=chain_length,
        sysenvcoupling=first(empty_chain_coups),
        environment_chain_frequencies=first(empty_chain_freqs, chain_length),
        environment_chain_couplings=first(empty_chain_coups[2:end], chain_length - 1),
        dt=parsedargs["time_step"],
        tmax=parsedargs["max_time"],
        maxbonddim=parsedargs["max_bond_dimension"],
        io_file=measurements_file,
        io_ranks=bonddims_file,
        io_times=simtime_file,
        operators=ops,
    )

    pack!(
        parsedargs["output"] * ".h5";
        argsdict=parsedargs,
        expvals_file=measurements_file,
        bonddimensions_file=bonddims_file,
        walltime_file=simtime_file,
    )

    return nothing
end

main()
