using ITensors, ITensorMPS, MPSTimeEvolution, CSV, MarkovianClosure
using Base.Iterators: peel

include("../shared_functions.jl")

function simulation(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcouplingL,
    sysenvcouplingR,
    nenvironment,
    environmentL_chain_frequencies,
    environmentL_chain_couplings,
    environmentR_chain_frequencies,
    environmentR_chain_couplings,
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

    # L : initially filled, R : initially empty
    environmentL = ModeChain(
        range(; start=2nsystem + 1, step=4, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )
    altenvironmentL = ModeChain(
        range(; start=2nsystem + 2, step=4, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )

    truncated_environmentL, mcL = markovianclosure(environmentL, nclosure, nenvironment)
    truncated_altenvironmentL, _ = markovianclosure(altenvironmentL, nclosure, nenvironment)

    environmentR = ModeChain(
        range(; start=2nsystem + 3, step=4, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )
    altenvironmentR = ModeChain(
        range(; start=2nsystem + 4, step=4, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )

    truncated_environmentR, mcR = markovianclosure(environmentR, nclosure, nenvironment)
    truncated_altenvironmentR, _ = markovianclosure(altenvironmentR, nclosure, nenvironment)

    environments_last_site = max(
        truncated_environmentL.range[end],
        truncated_altenvironmentL.range[end],
        truncated_environmentR.range[end],
        truncated_altenvironmentR.range[end],
    )

    closureL = ModeChain(
        range(; start=environments_last_site + 1, step=4, length=nclosure),
        freqs(mcL),
        # We hardcode the transformation from empty to filled MC, in which we assume that
        # the alpha coefficients are real. TODO find an automated way to do this.
        -innercoups(mcL),
    )
    altclosureL = ModeChain(
        range(; start=environments_last_site + 2, step=4, length=nclosure),
        freqs(mcL),
        -innercoups(mcL),
    )
    closureR = ModeChain(
        range(; start=environments_last_site + 3, step=4, length=nclosure),
        freqs(mcR),
        innercoups(mcR),
    )
    altclosureR = ModeChain(
        range(; start=environments_last_site + 4, step=4, length=nclosure),
        freqs(mcR),
        innercoups(mcR),
    )

    function starts_from_occ(n)
        return (
            in(n, system) ||
            in(n, altsystem) ||
            in(n, truncated_environmentL) ||
            in(n, truncated_altenvironmentL) ||
            in(n, closureL) ||
            in(n, altclosureL)
        )
    end

    site(tags) = addtags(siteind("Fermion"; conserve_nfparity=true), tags)
    sites = [
        site("System")
        site("System(alt)")
        interleave(
            [site("EnvL,n=$n") for n in 1:length(truncated_environmentL)],
            [site("EnvL(alt),n=$n") for n in 1:length(truncated_altenvironmentL)],
            [site("EnvR,n=$n") for n in 1:length(truncated_environmentR)],
            [site("EnvR(alt),n=$n") for n in 1:length(truncated_altenvironmentR)],
        )
        interleave(
            [site("ClosureL,n=$n") for n in 1:length(closureL)],
            [site("ClosureL(alt),n=$n") for n in 1:length(altclosureL)],
            [site("ClosureR,n=$n") for n in 1:length(closureR)],
            [site("ClosureR(alt),n=$n") for n in 1:length(altclosureR)],
        )
    ]

    @assert findall(idx -> hastags(idx, "EnvL"), sites) == truncated_environmentL.range
    @assert findall(idx -> hastags(idx, "EnvR"), sites) == truncated_environmentR.range
    @assert findall(idx -> hastags(idx, "ClosureL"), sites) == closureL.range
    @assert findall(idx -> hastags(idx, "ClosureR"), sites) == closureR.range

    st = SiteType("Fermion")
    initstate = MPS(sites, n -> starts_from_occ(n) ? "Occ" : "Emp")

    # -- Leftover unitary part -------------------------------------------------------------
    ad_h =
        spinchain(st, join(system, truncated_environmentL, sysenvcouplingL)) -
        spinchain(st, join(altsystem, truncated_altenvironmentL, sysenvcouplingR)) +
        spinchain(st, join(system, truncated_environmentR, sysenvcouplingR)) -
        spinchain(st, join(altsystem, truncated_altenvironmentR, sysenvcouplingR))
    # --------------------------------------------------------------------------------------

    # -- Initially filled MC ---------------------------------------------------------------
    # NN interactions
    ad_h += spinchain(st, closureL) - spinchain(st, altclosureL)
    # Interaction with last environment site
    for (z, site) in zip(conj.(outercoups(mcL)), closureL.range)
        #                ────    manual empty → filled transformation
        ad_h += z, "cdag", site, "c", last(truncated_environmentL.range)
        ad_h += conj(z), "cdag", last(truncated_environmentL.range), "c", site
    end
    for (z, site) in zip(conj.(outercoups(mcL)), altclosureL.range)
        #                ────    manual empty → filled transformation
        ad_h -= z, "cdag", site, "c", last(truncated_altenvironmentL.range)
        ad_h -= conj(z), "cdag", last(truncated_altenvironmentL.range), "c", site
    end
    # Dissipation operator
    DL = OpSum()
    for (g, site, altsite) in zip(damps(mcL), closureL.range, altclosureL.range)
        DL += g, "cdag", site, "cdag", altsite
        DL += -0.5g, "c", site, "cdag", site
        DL += -0.5g, "c", altsite, "cdag", altsite
    end
    # --------------------------------------------------------------------------------------

    # -- Initially empty MC ----------------------------------------------------------------
    # NN interactions
    ad_h += spinchain(st, closureR) - spinchain(st, altclosureR)
    # Interaction with last environment site
    for (z, site) in zip(outercoups(mcR), closureR.range)
        ad_h += z, "cdag", site, "c", last(truncated_environmentR.range)
        ad_h += conj(z), "cdag", last(truncated_environmentR.range), "c", site
    end
    for (z, site) in zip(outercoups(mcR), altclosureR.range)
        ad_h -= z, "cdag", site, "c", last(truncated_altenvironmentR.range)
        ad_h -= conj(z), "cdag", last(truncated_altenvironmentR.range), "c", site
    end
    # Dissipation operator
    DR = OpSum()
    for (g, site, altsite) in zip(damps(mcR), closureR.range, altclosureR.range)
        DR += -g, "c", site, "c", altsite
        DR += -0.5g, "cdag", site, "c", site
        DR += -0.5g, "cdag", altsite, "c", altsite
    end
    # --------------------------------------------------------------------------------------

    L = MPO(-im * ad_h + DL + DR, sites)

    cb = SuperfermionCallback(operators, sites, dt)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    state_t = enlargelinks(
        initstate, maxbonddim; ref_state=n -> starts_from_occ(n) ? "Occ" : "Emp"
    )

    simulation_files_info(;
        measurements_file=io_file, bonddims_file=io_ranks, simtime_file=io_times
    )

    tdvp1vec!(
        state_t,
        L,
        dt,
        tmax;
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

    empty_chainL_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqfilled"]
    sysenvcouplingL, empty_chainL_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupfilled"]
    )
    empty_chainR_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    sysenvcouplingR, empty_chainR_coups = peel(
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
        sysenvcouplingL=sysenvcouplingL,
        environmentL_chain_frequencies=empty_chainL_freqs,
        environmentL_chain_couplings=collect(empty_chainL_coups),
        sysenvcouplingR=sysenvcouplingR,
        environmentR_chain_frequencies=empty_chainR_freqs,
        environmentR_chain_couplings=collect(empty_chainR_coups),
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
