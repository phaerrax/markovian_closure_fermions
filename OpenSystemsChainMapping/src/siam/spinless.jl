function siam_spinless_pure_state(;
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
    maxbonddim,
)
    system = ModeChain(1:nsystem, [system_energy], [])
    environmentL = ModeChain(
        range(; start=nsystem + 1, step=2, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )
    environmentR = ModeChain(
        range(; start=nsystem + 2, step=2, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )
    environmentL = first(environmentL, nenvironment)
    environmentR = first(environmentR, nenvironment)

    function init(n)
        return if in(n, system)
            system_initial_state
        elseif in(n, environmentL)
            "Occ"
        else
            "Emp"
        end
    end
    st = SiteType("Fermion")
    site(tags) = addtags(siteind("Fermion"), tags)
    sites = [
        site("System")
        interleave(
            [site("EnvL") for n in 1:nenvironment], [site("EnvR") for n in 1:nenvironment]
        )
    ]
    for n in eachindex(sites)
        sites[n] = addtags(sites[n], "n=$n")
    end
    initstate = MPS(sites, init)

    @assert findall(idx -> hastags(idx, "System"), sites) == system.range
    @assert findall(idx -> hastags(idx, "EnvL"), sites) == environmentL.range

    h = spinchain(
        SiteType("Fermion"),
        join(
            reverse(environmentR),
            join(system, environmentL, sysenvcouplingL),
            sysenvcouplingR,
        ),
    )
    H = MPO(h, sites)

    initstate = enlargelinks(initstate, maxbonddim; ref_state=init)

    return initstate, H
end

function siam_spinless_superfermions_mc(;
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

    truncated_environmentL, mcL = markovianclosure(environmentL, nclosure, nenvironment)
    truncated_altenvironmentL, _ = markovianclosure(
        altenvironmentL, nclosure, nenvironment;
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

    truncated_environmentR, mcR = markovianclosure(environmentR, nclosure, nenvironment)
    truncated_altenvironmentR, _ = markovianclosure(
        altenvironmentR, nclosure, nenvironment;
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

    function init(n)
        return if in(n, system) || in(n, altsystem)
            system_initial_state
        elseif (
            in(n, truncated_environmentL) ||
            in(n, truncated_altenvironmentL) ||
            in(n, closureL) ||
            in(n, altclosureL)
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
    initstate = MPS(sites, init)

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
    initstate = enlargelinks(initstate, maxbonddim; ref_state=init)

    return initstate, L
end

function siam_spinless_vectorised_mc(;
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
    system = ModeChain(range(; start=1, step=1, length=nsystem), [system_energy], [])

    # L : initially filled, R : initially empty
    environmentL = ModeChain(
        range(; start=nsystem + 1, step=2, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )
    environmentR = ModeChain(
        range(; start=nsystem + 2, step=2, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )

    truncated_environmentL, mcL = markovianclosure(environmentL, nclosure, nenvironment)
    truncated_environmentR, mcR = markovianclosure(environmentR, nclosure, nenvironment)

    environments_last_site = maximum(
        [truncated_environmentL.range; truncated_environmentR.range]
    )

    closureL = ModeChain(
        range(; start=environments_last_site + 1, step=2, length=nclosure),
        freqs(mcL),
        # We hardcode the transformation from empty to filled MC, in which we assume that
        # the alpha coefficients are real. TODO find an automated way to do this.
        -innercoups(mcL),
    )
    closureR = ModeChain(
        range(; start=environments_last_site + 2, step=2, length=nclosure),
        freqs(mcR),
        innercoups(mcR),
    )

    function init(n)
        return if in(n, system)
            system_initial_state
        elseif in(n, truncated_environmentL) || in(n, closureL)
            "Occ"
        else
            "Emp"
        end
    end

    st = SiteType("vFermion")
    site(tags) = addtags(siteind("vFermion"), tags)
    sites = [
        site("System")
        interleave(
            [site("EnvL") for n in 1:nenvironment], [site("EnvR") for n in 1:nenvironment]
        )
        interleave(
            [site("ClosureL") for n in 1:nclosure], [site("ClosureR") for n in 1:nclosure]
        )
    ]
    for n in eachindex(sites)
        sites[n] = addtags(sites[n], "n=$n")
    end
    initstate = MPS(sites, init)

    @assert findall(idx -> hastags(idx, "System"), sites) == system.range
    @assert findall(idx -> hastags(idx, "EnvL"), sites) == truncated_environmentL.range
    @assert findall(idx -> hastags(idx, "EnvR"), sites) == truncated_environmentR.range
    @assert findall(idx -> hastags(idx, "ClosureL"), sites) == closureL.range
    @assert findall(idx -> hastags(idx, "ClosureR"), sites) == closureR.range

    ad_h = OpSum()
    DL = OpSum()
    DR = OpSum()

    ad_h = spinchain(
        SiteType("vFermion"),
        join(
            reverse(truncated_environmentR),
            join(system, truncated_environmentL, sysenvcouplingL),
            sysenvcouplingR,
        ),
    )

    # -- Initially filled MC ---------------------------------------------------------------
    # NN interactions
    ad_h_mcL = spinchain(st, closureL)
    # Interaction with last environment site
    chain_edge_site = last(truncated_environmentL.range)
    for (n, z) in zip(closureL.range, outercoups(mcL))
        jws = jwstring(; start=chain_edge_site, stop=n)
        ad_h_mcL += (
            z * gkslcommutator("A†", chain_edge_site, jws..., "A", n) +
            conj(z) * gkslcommutator("A", chain_edge_site, jws..., "A†", n)
        )
    end
    # Dissipation operator
    for (n, g) in zip(closureL.range, damps(mcL))
        # a† ρ a
        opstring = [repeat(["F⋅ * ⋅F"], n - 1); "A†⋅ * ⋅A"]
        DL += (g, interleave(opstring, 1:n)...)
        # -0.5 (a a† ρ + ρ a a†) = 0.5 (a† a ρ + ρ a† a) - ρ
        DL += 0.5g, "N⋅", n
        DL += 0.5g, "⋅N", n
        DL += -g, "Id", n
    end
    # --------------------------------------------------------------------------------------

    # -- Initially empty MC ----------------------------------------------------------------
    # NN interactions
    ad_h_mcR = spinchain(st, closureR)
    # Interaction with last environment site
    chain_edge_site = last(truncated_environmentR.range)
    for (n, z) in zip(closureR.range, outercoups(mcR))
        jws = jwstring(; start=chain_edge_site, stop=n)
        ad_h_mcR += (
            conj(z) * gkslcommutator("A†", chain_edge_site, jws..., "A", n) +
            z * gkslcommutator("A", chain_edge_site, jws..., "A†", n)
        )
    end
    # Dissipation operator
    for (n, g) in zip(closureR.range, damps(mcR))
        # a ρ a†
        opstring = [repeat(["F⋅ * ⋅F"], n - 1); "A⋅ * ⋅A†"]
        DR += (g, interleave(opstring, 1:n)...)
        # -0.5 (a† a ρ + ρ a† a)
        DR += -0.5g, "N⋅", n
        DR += -0.5g, "⋅N", n
    end
    # --------------------------------------------------------------------------------------

    L = MPO(ad_h + ad_h_mcL + ad_h_mcR + DL + DR, sites)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    growMPS!(initstate, maxbonddim)

    return initstate, L
end
