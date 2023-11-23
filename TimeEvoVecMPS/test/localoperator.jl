using ITensors
using PseudomodesTTEDOPA
using TimeEvoVecMPS

let
    system_length = 1

    chain_length = 10
    coups = [0.25ℯ^(-2n) for n in 1:10]
    freqs = [0.5 + ℯ^(-4n) for n in 1:10]
    chain_range = system_length .+ (1:chain_length)

    n_sites = chain_length + system_length
    sites = siteinds("vFermion", n_sites)
    vecρₜ = MPS(sites, [i == 1 ? "Occ" : "Emp" for i in 1:n_sites])

    # Master equation
    ℓ =
        gkslcommutator("N", 1) +
        exchange_interaction(sites[1], sites[chain_range[1]]; coupling_constant=coups[1]) +
        spin_chain(freqs[1:chain_length], coups[2:chain_length], sites[chain_range])

    L = MPO(ℓ, sites)

    # Enlarge the bond dimensions so that TDVP1 has the possibility to grow
    # the number of singular values between the bonds.
    growMPS!(vecρₜ, 10)

    timestep = 0.01
    tmax = 1

    operators = [
        LocalOperator(["vAdag", "vA"], 1),
        LocalOperator(["vA", "vAdag"], 2),
        LocalOperator(["vN"], 1),
    ]
    cb = LocalOperatorCallback(operators, sites, 5 * timestep)

    tmpfile = tempname()
    tdvp1vec!(
        vecρₜ,
        L,
        timestep,
        tmax,
        sites;
        hermitian=false,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    results = open(tmpfile, "r") do test_res
        read(test_res, String)
    end
    print(results)

    return nothing
end
