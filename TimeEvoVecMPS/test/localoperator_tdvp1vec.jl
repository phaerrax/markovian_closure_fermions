using ITensors
using PseudomodesTTEDOPA
using TimeEvoVecMPS

let
    system_length = 1

    chain_length = 10
    coups = [0.25/n for n in 1:10]
    freqs = [0.5 /n for n in 1:10]
    chain_range = system_length .+ (1:chain_length)

    n_sites = chain_length + system_length
    sites = siteinds("vOsc", n_sites; dim=4)
    vecρₜ = MPS(sites, [i == 1 ? "1" : "0" for i in 1:n_sites])

    # Master equation
    ℓ = gkslcommutator("N", 1) + coups[1] * gkslcommutator( "Asum", 1, "Asum", 2)
    for i in 1:(chain_length-1)
        ℓ += freqs[i] * gkslcommutator("N", chain_range[i])
        ℓ += coups[i+1] * gkslcommutator("A", chain_range[i], "Adag", chain_range[i+1])
        ℓ += coups[i+1] * gkslcommutator("Adag", chain_range[i], "A", chain_range[i+1])
        end
        ℓ += freqs[chain_length]* gkslcommutator( "N", chain_range[chain_length])

    L = MPO(ℓ, sites)

    # Enlarge the bond dimensions so that TDVP1 has the possibility to grow
    # the number of singular values between the bonds.
    growMPS!(vecρₜ, 20)

    timestep = 0.02
    tmax = 10

    operators = [
        LocalOperator(Dict(1 => "vAdag", 2 => "vA")),
        LocalOperator(Dict(3 => "vA", 5 => "vAdag")),
        LocalOperator(Dict(4 => "vN")),
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
        progress=false,
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
