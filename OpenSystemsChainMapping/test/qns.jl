using ITensors: hasqns
using ITensorMPS: linkdims
using MPSTimeEvolution: enlargelinks

# Create a very simple spinless SIAM and check that QNs have been activated
function siam_spinless_pure_state_hasqns(; N=5, maxdim=10)
    initstate, _ = siam_spinless_pure_state(;
        nsystem=1,
        system_energy=-1 / 2,
        system_initial_state="Occ",
        nenvironment=5,
        sysenvcouplingL=1 / 4,
        environmentL_chain_frequencies=fill(1, N),
        environmentL_chain_couplings=fill(1 / 3, N - 1),
        sysenvcouplingR=1 / 4,
        environmentR_chain_frequencies=fill(1, N),
        environmentR_chain_couplings=fill(-1 / 3, N - 1),
        maxbonddim=maxdim,
        conserve_nf=true,
        conserve_nfparity=true,
    )

    return all(hasqns.(initstate))
end

function siam_spinless_superfermions_mc_hasqns(; nenvironment=2, nclosure=6, maxdim=10)
    N = nenvironment + nclosure
    initstate, _ = siam_spinless_superfermions_mc(;
        nsystem=1,
        system_energy=1 / 2,
        system_initial_state="Occ",
        nenvironment=5,
        sysenvcouplingL=4,
        environmentL_chain_frequencies=fill(1, N),
        environmentL_chain_couplings=fill(3, N - 1),
        sysenvcouplingR=4,
        environmentR_chain_frequencies=fill(1, N),
        environmentR_chain_couplings=fill(3, N - 1),
        nclosure=6,
        maxbonddim=maxdim,
        conserve_nfparity=true,
    )

    return all(hasqns.(initstate))
end

# Check that the bond dimensions of the initial states can be increased
function siam_spinless_pure_state_qns_enlarge(; N=20, maxdim=5)
    # Make sure that N is large enough...
    initstate, _ = siam_spinless_pure_state(;
        nsystem=1,
        system_energy=-1 / 2,
        system_initial_state="Occ",
        nenvironment=5,
        sysenvcouplingL=1 / 4,
        environmentL_chain_frequencies=fill(1, N),
        environmentL_chain_couplings=fill(1 / 3, N - 1),
        sysenvcouplingR=1 / 4,
        environmentR_chain_frequencies=fill(1, N),
        environmentR_chain_couplings=fill(-1 / 3, N - 1),
        maxbonddim=maxdim,
        conserve_nf=true,
        conserve_nfparity=true,
    )

    # Exclude edge sites because the bond dimension is always naturally lower at the edges.
    return all(linkdims(initstate)[3:(end - 2)] .== maxdim)
end

function siam_spinless_superfermions_mc_qns_enlarge(; nenvironment=4, nclosure=6, maxdim=5)
    N = nenvironment + nclosure
    initstate, _ = siam_spinless_superfermions_mc(;
        nsystem=1,
        system_energy=-1 / 2,
        system_initial_state="Occ",
        nenvironment=5,
        sysenvcouplingL=1 / 4,
        environmentL_chain_frequencies=fill(1, N),
        environmentL_chain_couplings=fill(1 / 3, N - 1),
        sysenvcouplingR=1 / 4,
        environmentR_chain_frequencies=fill(1, N),
        environmentR_chain_couplings=fill(-1 / 3, N - 1),
        nclosure=6,
        maxbonddim=maxdim,
        conserve_nfparity=true,
    )

    # Exclude edge sites because the bond dimension is always naturally lower at the edges.
    return all(linkdims(initstate)[3:(end - 2)] .== maxdim)
end
