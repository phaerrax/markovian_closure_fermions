zerosite!(PH::ProjMPO) = (PH.nsite = 0)
singlesite!(PH::ProjMPO) = (PH.nsite = 1)
twosite!(PH::ProjMPO) = (PH.nsite = 2)

struct TDVP1 end
struct TDVP2 end

"""
    writeheaders_data(io_file, cb; kwargs...)

Prepare the output file `io_file`, writing the column headers for storing the data of
the observables defined in `cb`, the time steps, and other basic quantities.
"""
function writeheaders_data(io_file, cb; kwargs...)
    io_handle = nothing
    if !isnothing(io_file)
        io_handle = open(io_file, "w")
        @printf(io_handle, "%20s", "time")
        res = measurements(cb)
        for o in sort(collect(keys(res)))
            @printf(io_handle, "%40s", o)
        end
        if get(kwargs, :store_psi0, false)
            @printf(io_handle, "%40s%40s", "re_over", "im_over")
        end
        @printf(io_handle, "%40s", "Norm")
        @printf(io_handle, "\n")
    end

    return io_handle
end

"""
    writeheaders_ranks(ranks_file, N)

Prepare the output file `ranks_file`, writing the column headers for storing the data
relative to the ranks of a MPS of the given length `N`.
"""
function writeheaders_ranks(ranks_file, N)
    ranks_handle = nothing
    if !isnothing(ranks_file)
        ranks_handle = open(ranks_file, "w")
        @printf(ranks_handle, "%20s", "time")
        for r in 1:(N - 1)
            @printf(ranks_handle, "%10d", r)
        end
        @printf(ranks_handle, "\n")
    end

    return ranks_handle
end

"""
    writeheaders_stime(times_file)

Prepare the output file `times_file`, writing the column headers for the simulation
time data.
"""
function writeheaders_stime(times_file)
    times_handle = nothing
    if !isnothing(times_file)
        times_handle = open(times_file, "w")
        @printf(times_handle, "%20s", "walltime (sec)")
        @printf(times_handle, "\n")
    end

    return times_handle
end

function printoutput_data(io_handle, cb, psi; kwargs...)
    if !isnothing(io_handle)
        results = measurements(cb)
        @printf(io_handle, "%40.15f", measurement_ts(cb)[end])
        for o in sort(collect(keys(results)))
            @printf(io_handle, "%40.15f", results[o][end][1])
        end

        if get(kwargs, :store_psi0, false)
            psi0 = get(kwargs, :psi0, nothing)
            overlap = dot(psi0, psi)
            @printf(io_handle, "%40.15f%40.15f", real(overlap), imag(overlap))
        end

        # Print the norm of the trace of the state, depending on whether the MPS represents
        # a pure state or a vectorized density matrix.
        isvectorized = get(kwargs, :vectorized, false)
        if isvectorized
            @printf(io_handle, "%40.15f", real(inner(MPS(kwargs[:sites], "vecId"), psi)))
        else
            @printf(io_handle, "%40.15f", norm(psi))
        end
        @printf(io_handle, "\n")
        flush(io_handle)
    end

    return nothing
end

function printoutput_ranks(ranks_handle, cb, state)
    if !isnothing(ranks_handle)
        @printf(ranks_handle, "%40.15f", measurement_ts(cb)[end])

        for bonddim in ITensors.linkdims(state)
            @printf(ranks_handle, "%10d", bonddim)
        end

        @printf(ranks_handle, "\n")
        flush(ranks_handle)
    end

    return nothing
end

function printoutput_stime(times_handle, stime::Real)
    if !isnothing(times_handle)
        @printf(times_handle, "%20.4f\n", stime)
        flush(times_handle)
    end

    return nothing
end
