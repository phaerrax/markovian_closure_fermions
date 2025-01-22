using ITensors, ITensorMPS, MPSTimeEvolution, CSV, MarkovianClosure
using Base.Iterators: peel

include("../../shared_functions.jl")

function MPSTimeEvolution.enlargelinks(v, dims::Vector{<:Integer}; ref_state=nothing)
    diff_linkdims = max.(dims .- linkdims(v), 1)
    x = if hasqns(first(v))
        if isnothing(ref_state)
            error("Initial state required to use enlargelinks with QNs")
        else
            random_mps(siteinds(v), ref_state; linkdims=diff_linkdims)
        end
    else
        random_mps(siteinds(v); linkdims=diff_linkdims)
    end
    orthogonalize!(x, 1)
    v_ext = add(orthogonalize(v, 1), 0 * x; alg="directsum")
    return v_ext
end

function MPSTimeEvolution.enlargelinks(v, dim::Integer; kwargs...)
    return enlargelinks(v, fill(dim, length(v) - 1); kwargs...)
end

function siam_spinless_superfermions_2env_mc(;
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
    maxbonddim,
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

    truncated_environmentL, mcL = markovianclosure(
        environmentL, nclosure, nenvironment; asymptoticfrequency=0, asymptoticcoupling=0.5
    )
    truncated_altenvironmentL, _ = markovianclosure(
        altenvironmentL,
        nclosure,
        nenvironment;
        asymptoticfrequency=0,
        asymptoticcoupling=0.5,
    )

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

    truncated_environmentR, mcR = markovianclosure(
        environmentR, nclosure, nenvironment; asymptoticfrequency=0, asymptoticcoupling=0.5
    )
    truncated_altenvironmentR, _ = markovianclosure(
        altenvironmentR,
        nclosure,
        nenvironment;
        asymptoticfrequency=0,
        asymptoticcoupling=0.5,
    )

    environments_last_site = maximum(
        [
            truncated_environmentL.range
            truncated_altenvironmentL.range
            truncated_environmentR.range
            truncated_altenvironmentR.range
        ],
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

    function initstate_labels(n)
        return if (n in system || n in altsystem)
            system_initial_state
        elseif (
            n in truncated_environmentL ||
            n in truncated_altenvironmentL ||
            n in closureL ||
            n in altclosureL
        )
            "Occ"
        else
            "Emp"
        end
    end

    site(tags) = addtags(siteind("Fermion"; conserve_nfparity=true), tags)
    sites = [
        site("System")
        site("System(alt)")
        interleave(
            [site("EnvL,n=$n") for n in 1:nenvironment],
            [site("EnvL(alt),n=$n") for n in 1:nenvironment],
            [site("EnvR,n=$n") for n in 1:nenvironment],
            [site("EnvR(alt),n=$n") for n in 1:nenvironment],
        )
        interleave(
            [site("ClosureL,n=$n") for n in 1:nclosure],
            [site("ClosureL(alt),n=$n") for n in 1:nclosure],
            [site("ClosureR,n=$n") for n in 1:nclosure],
            [site("ClosureR(alt),n=$n") for n in 1:nclosure],
        )
    ]

    @assert findall(idx -> hastags(idx, "EnvL"), sites) == truncated_environmentL.range
    @assert findall(idx -> hastags(idx, "EnvR"), sites) == truncated_environmentR.range
    @assert findall(idx -> hastags(idx, "ClosureL"), sites) == closureL.range
    @assert findall(idx -> hastags(idx, "ClosureR"), sites) == closureR.range

    st = SiteType("Fermion")
    initstate = MPS(sites, initstate_labels)

    ad_h = OpSum()
    DL = OpSum()
    DR = OpSum()

    # -- Leftover unitary part -------------------------------------------------------------
    ad_h +=
        spinchain(
            st,
            join(
                reverse(truncated_environmentL),
                join(system, truncated_environmentR, sysenvcouplingR),
                sysenvcouplingL,
            ),
        ) - spinchain(
            st,
            join(
                reverse(truncated_altenvironmentL),
                join(altsystem, truncated_altenvironmentR, sysenvcouplingR),
                sysenvcouplingL,
            ),
        )
    # --------------------------------------------------------------------------------------

    # -- Initially filled MC ---------------------------------------------------------------
    # NN interactions
    ad_h += spinchain(st, closureL) - spinchain(st, altclosureL)
    # Interaction with last environment n
    for (z, n) in zip(outercoups(mcL), closureL.range)
        ad_h += z, "cdag", n, "c", last(truncated_environmentL.range)
        ad_h += conj(z), "cdag", last(truncated_environmentL.range), "c", n
    end
    for (z, n) in zip(outercoups(mcL), altclosureL.range)
        ad_h -= conj(z), "cdag", n, "c", last(truncated_altenvironmentL.range)
        ad_h -= z, "cdag", last(truncated_altenvironmentL.range), "c", n
    end
    # Dissipation operator
    for (g, n, altn) in zip(damps(mcL), closureL.range, altclosureL.range)
        DL += g, "cdag", n, "cdag", altn
        DL += -0.5g, "c", n, "cdag", n
        DL += -0.5g, "c", altn, "cdag", altn
    end
    # --------------------------------------------------------------------------------------

    # -- Initially empty MC ----------------------------------------------------------------
    # NN interactions
    ad_h += spinchain(st, closureR) - spinchain(st, altclosureR)
    # Interaction with last environment site
    for (z, n) in zip(outercoups(mcR), closureR.range)
        ad_h += z, "cdag", n, "c", last(truncated_environmentR.range)
        ad_h += conj(z), "cdag", last(truncated_environmentR.range), "c", n
    end
    for (z, n) in zip(outercoups(mcR), altclosureR.range)
        ad_h -= conj(z), "cdag", n, "c", last(truncated_altenvironmentR.range)
        ad_h -= z, "cdag", last(truncated_altenvironmentR.range), "c", n
    end
    # Dissipation operator
    for (g, n, altn) in zip(damps(mcR), closureR.range, altclosureR.range)
        DR += -g, "c", n, "c", altn
        DR += -0.5g, "cdag", n, "c", n
        DR += -0.5g, "cdag", altn, "c", altn
    end
    # --------------------------------------------------------------------------------------

    L = MPO(-im * ad_h + DL + DR, sites)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    initstate = enlargelinks(initstate, maxbonddim; ref_state=initstate_labels)

    return initstate, L
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

    initstate, L = siam_spinless_superfermions_2env_mc(;
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
