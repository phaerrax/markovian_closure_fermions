using ITensors, ITensorMPS
using LindbladVectorizedTensors
using TimeEvoVecMPS

using IterTools: partition

let
    n_sites = 10
    sites = siteinds("S=1/2", n_sites)
    freqs = [1 for s in sites]
    coups = [0.5 for s in sites]

    psiL_t = randomMPS(sites; linkdims=10)
    psiR_t = apply(op("S+", sites, 1), psiL_t)

    # Schrödinger equation
    h = OpSum()
    for (i, j) in partition(eachindex(sites), 2, 1)
        h += coups[i], "S-", i, "S+", j
        h += coups[i], "S+", i, "S-", j
    end
    for i in eachindex(sites)
        h += freqs[i], "Z", i
    end

    H = MPO(h, sites)

    timestep = 0.1
    tmax = 10

    operators = [
        LocalOperator(Dict(1 => "Z"))
        LocalOperator(Dict(1 => "S+"))
    ]
    cb = ExpValueCallback(operators, sites, 5 * timestep)

    tmpfile_data = tempname()
    tmpfile_ranks = tempname()
    tmpfile_times = tempname()

    jointtdvp1!(
        (psiR_t, psiL_t),
        H,
        timestep,
        tmax;
        hermitian=true,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=tmpfile_data,
        io_ranks=tmpfile_ranks,
        io_times=tmpfile_times,
    )

    @info "Measurements and norms"
    results = open(tmpfile_data, "r") do test_res
        read(test_res, String)
    end
    print(results)

    @info "Bond dimensions"
    ranks = open(tmpfile_ranks, "r") do test_res
        read(test_res, String)
    end
    print(ranks)

    @info "Simulation time"
    times = open(tmpfile_times, "r") do test_res
        read(test_res, String)
    end
    print(times)

    return nothing
end
