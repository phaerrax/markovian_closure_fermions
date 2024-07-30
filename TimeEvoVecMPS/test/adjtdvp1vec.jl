using ITensors
using LindbladVectorizedTensors
using TimeEvoVecMPS

let
    nsites = 10
    sites = siteinds("vS=1/2", nsites)
    freqs = fill(1, nsites)
    coups = fill(0.5, nsites - 1)

    initialstates = [
        MPS(sites, "Dn"),
        MPS(sites, [isodd(n) ? "Up" : "Dn" for n in 1:nsites]),
        MPS(sites, [iseven(n) ? "Up" : "Dn" for n in 1:nsites]),
    ]

    # Evolution operator
    adjℓ = spin_chain_adjoint(freqs, coups, sites)
    adjL = MPO(adjℓ, sites)

    timestep = 0.1
    tmax = 10
    meas_stride = timestep * 5

    tmpfile_data = tempname()
    tmpfile_ranks = tempname()
    tmpfile_times = tempname()

    operator = MPS(ComplexF64, sites, [n == 1 ? "vZ" : "vId" for n in 1:nsites])

    adjtdvp1vec!(
        operator,
        initialstates,
        adjL,
        timestep,
        tmax,
        meas_stride,
        sites;
        hermitian=false,
        normalize=false,
        progress=true,
        store_psi0=false,
        io_file=tmpfile_data,
        io_ranks=tmpfile_ranks,
        io_times=tmpfile_times,
        initialstatelabels=["dn", "odd", "even"],
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
