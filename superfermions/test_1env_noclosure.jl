using ITensors, ITensorMPS, MPSTimeEvolution, CSV
using Base.Iterators: peel

include("../shared_functions.jl")

function siam_spinless_superfermions_1env(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcoupling,
    nenvironment,
    environment_chain_frequencies,
    environment_chain_couplings,
    maxbonddim,
)
    sites = siteinds("Fermion", 2nsystem + 2nenvironment; conserve_nfparity=true)
    system = ModeChain(range(; start=1, step=2, length=nsystem), [system_energy], [])
    altsystem = ModeChain(range(; start=2, step=2, length=nsystem), [system_energy], [])

    environment = ModeChain(
        range(; start=2nsystem + 1, step=2, length=nenvironment),
        first(environment_chain_frequencies, nenvironment),
        first(environment_chain_couplings, nenvironment - 1),
    )
    altenvironment = ModeChain(
        range(; start=2nsystem + 2, step=2, length=nenvironment),
        first(environment_chain_frequencies, nenvironment),
        first(environment_chain_couplings, nenvironment - 1),
    )

    initstate = MPS(sites, n -> n ≤ 2nsystem ? "Occ" : "Emp")

    ad_h =
        spinchain(SiteType("Fermion"), join(system, environment, sysenvcoupling)) -
        spinchain(SiteType("Fermion"), join(altsystem, altenvironment, sysenvcoupling))

    L = MPO(-im * ad_h, sites)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    state_t = enlargelinks(
        initstate, maxbonddim; ref_state=n -> n ≤ 2nsystem ? "Occ" : "Emp"
    )

    return initstate, L
end

function main()
    parsedargs = parsecommandline()

    empty_chain_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    sysenvcoupling, empty_chain_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]
    )

    measurements_file = parsedargs["output"] * "_measurements.csv"
    bonddims_file = parsedargs["output"] * "_bonddims.csv"
    simtime_file = parsedargs["output"] * "_simtime.csv"

    initstate, L = siam_spinless_superfermions_1env(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=parsedargs["environment_sites"],
        sysenvcoupling=sysenvcoupling,
        environment_chain_frequencies=empty_chain_freqs,
        environment_chain_couplings=collect(empty_chain_coups),
        maxbonddim=parsedargs["max_bond_dimension"],
    )

    dt = parsedargs["time_step"]
    tmax = parsedargs["max_time"]
    operators = parseoperators(parsedargs["observables"])
    cb = SuperfermionCallback(operators, siteinds(initstate), dt)

    simulation_files_info(;
        measurements_file=measurements_file,
        bonddims_file=bonddims_file,
        simtime_file=simtime_file,
    )

    tdvp1vec!(
        initstate,
        L,
        dt,
        tmax;
        callback=cb,
        io_file=measurements_file,
        io_ranks=bonddims_file,
        io_times=simtime_file,
        superfermions=true,
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
