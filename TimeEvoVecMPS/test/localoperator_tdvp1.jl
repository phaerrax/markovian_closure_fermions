using ITensors, ITensorMPS
using LindbladVectorizedTensors
using TimeEvoVecMPS

let
    system_length = 1

    chain_length = 10
    coups = [0.25 / n for n in 1:10]
    freqs = [0.5 / n for n in 1:10]
    chain_range = system_length .+ (1:chain_length)

    n_sites = chain_length + system_length
    sites = siteinds("Osc", n_sites; dim=4)
    psi_t = MPS(sites, [i == 1 ? "1" : "0" for i in 1:n_sites])

    # Schrödinger equation
    h = OpSum()
    h += "N", 1
    h += coups[1], "Asum", 1, "Asum", 2
    for i in 1:(chain_length - 1)
        h += freqs[i], "N", chain_range[i]
        h += coups[i + 1], "A", chain_range[i], "Adag", chain_range[i + 1]
        h += coups[i + 1], "Adag", chain_range[i], "A", chain_range[i + 1]
    end
    h += freqs[chain_length], "N", chain_range[chain_length]

    H = MPO(h, sites)

    # Enlarge the bond dimensions so that TDVP1 has the possibility to grow
    # the number of singular values between the bonds.
    growMPS!(psi_t, 10)

    timestep = 0.02
    tmax = 10

    operators = [
        LocalOperator(Dict(1 => "Adag", 2 => "A")),
        LocalOperator(Dict(1 => "A", 2 => "Adag")),
        LocalOperator(Dict(3 => "A", 5 => "Adag")),
        LocalOperator(Dict(4 => "N")),
    ]
    cb = ExpValueCallback(operators, sites, 5 * timestep)

    tmpfile = tempname()
    tdvp1!(
        psi_t,
        H,
        timestep,
        tmax;
        hermitian=true,
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
