using ITensors, ITensorMPS, MPSTimeEvolution, CSV, MarkovianClosure
using Base.Iterators: peel

include("../shared_functions.jl")

function simulation(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcoupling,
    nenvironment,
    environment_chain_frequencies,
    environment_chain_couplings,
    nclosure,
    dt,
    tmax,
    maxbonddim,
    io_file=nullfile(),
    io_ranks=nullfile(),
    io_times=nullfile(),
    operators,
)
    system = ModeChain(range(; start=1, step=2, length=nsystem), [system_energy], [])
    altsystem = ModeChain(range(; start=2, step=2, length=nsystem), [system_energy], [])

    environment = ModeChain(
        range(; start=2nsystem + 1, step=2, length=length(environment_chain_frequencies)),
        environment_chain_frequencies,
        environment_chain_couplings,
    )
    altenvironment = ModeChain(
        range(; start=2nsystem + 2, step=2, length=length(environment_chain_frequencies)),
        environment_chain_frequencies,
        environment_chain_couplings,
    )

    truncated_environment, mc = markovianclosure(environment, nclosure, nenvironment)
    truncated_altenvironment, _ = markovianclosure(altenvironment, nclosure, nenvironment)

    closure = ModeChain(
        range(; start=2nsystem + 2nenvironment + 1, step=2, length=nclosure),
        freqs(mc),
        innercoups(mc),
    )
    altclosure = ModeChain(
        range(; start=2nsystem + 2nenvironment + 2, step=2, length=nclosure),
        freqs(mc),
        innercoups(mc),
    )

    sites = siteinds(
        "Fermion", 2nsystem + 2nenvironment + 2nclosure; conserve_nfparity=true
    )
    initstate = MPS(sites, n -> n ≤ 2nsystem ? "Occ" : "Emp")

    ad_h =
        spinchain(
            SiteType("Fermion"), join(system, truncated_environment, sysenvcoupling)
        ) - spinchain(
            SiteType("Fermion"), join(altsystem, truncated_altenvironment, sysenvcoupling)
        ) + spinchain(SiteType("Fermion"), closure) -
        spinchain(SiteType("Fermion"), altclosure)

    # Interaction with last environment site
    for (z, site) in zip(outercoups(mc), closure.range)
        ad_h += z, "cdag", site, "c", last(truncated_environment.range)
        ad_h += conj(z), "cdag", last(truncated_environment.range), "c", site
    end
    for (z, site) in zip(outercoups(mc), altclosure.range)
        ad_h -= z, "cdag", site, "c", last(truncated_altenvironment.range)
        ad_h -= conj(z), "cdag", last(truncated_altenvironment.range), "c", site
    end

    # Dissipation operators
    D = OpSum()
    for (g, site, altsite) in zip(damps(mc), closure.range, altclosure.range)
        D += -g, "c", site, "c", altsite
        D += -0.5g, "cdag", site, "c", site
        D += -0.5g, "cdag", altsite, "c", altsite
    end

    L = MPO(-im * ad_h + D, sites)

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

    return state_t
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

    simulation(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=parsedargs["environment_sites"],
        nclosure=parsedargs["closure_sites"],
        sysenvcoupling=sysenvcoupling,
        environment_chain_frequencies=empty_chain_freqs,
        environment_chain_couplings=collect(empty_chain_coups),
        dt=parsedargs["time_step"],
        tmax=parsedargs["max_time"],
        maxbonddim=parsedargs["max_bond_dimension"],
        io_file=measurements_file,
        io_ranks=bonddims_file,
        io_times=simtime_file,
        operators=parseoperators(parsedargs["observables"]),
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
