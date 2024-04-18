export append_if_not_null, meanordefault

zerosite!(PH::ProjMPO) = (PH.nsite = 0)
singlesite!(PH::ProjMPO) = (PH.nsite = 1)
twosite!(PH::ProjMPO) = (PH.nsite = 2)

abstract type TDVP end
struct TDVP1 <: TDVP end
struct TDVP1vec <: TDVP end
struct TDVP2 <: TDVP end

"""
    meanordefault(v, default=nothing)

Compute the mean of `v`, unless `default` is given, in which case return `default`.
"""
function meanordefault(v, default=nothing)
    if isnothing(default)
        return sum(v) / length(v)
    else
        return default
    end
end

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
        for op in sort(collect(keys(res)))
            @printf(io_handle, "%40s%40s", name(op) * "_re", name(op) * "_im")
        end
        if get(kwargs, :store_psi0, false)
            @printf(io_handle, "%40s%40s", "overlap_re", "overlap_im")
        end
        @printf(io_handle, "%40s", "Norm")
        @printf(io_handle, "\n")
    end

    return io_handle
end

function writeheaders_data_double(io_file, cb; kwargs...)
    io_handle = nothing
    if !isnothing(io_file)
        io_handle = open(io_file, "w")
        @printf(io_handle, "%20s", "time")
        res = measurements(cb)
        for op in sort(collect(keys(res)))
            @printf(io_handle, "%40s%40s", name(op) * "_re", name(op) * "_im")
        end
        @printf(io_handle, "%40s%40s", "overlap12_re", "overlap12_im")
        if get(kwargs, :store_psi0, false)
            @printf(io_handle, "%40s%40s", "overlap1init_re", "overlap1init_im")
            @printf(io_handle, "%40s%40s", "overlap2init_re", "overlap2init_im")
        end
        @printf(io_handle, "%40s", "Norm1")
        @printf(io_handle, "%40s", "Norm2")
        @printf(io_handle, "\n")
    end

    return io_handle
end

"""
    writeheaders_ranks(ranks_file, N)

Prepare the output file `ranks_file`, writing the column headers for storing the data
relative to the ranks of a MPS of the given length `N`.
"""
function writeheaders_ranks(ranks_file, Ns::Int...)
    ranks_handle = nothing
    if !isnothing(ranks_file)
        ranks_handle = open(ranks_file, "w")
        @printf(ranks_handle, "%20s", "time")
        for N in Ns
            for r in 1:(N - 1)
                @printf(ranks_handle, "%10d", r)
            end
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

function printoutput_data(io_handle, cb, psi::MPS; kwargs...)
    if !isnothing(io_handle)
        results = measurements(cb)
        @printf(io_handle, "%40.15f", measurement_ts(cb)[end])
        for opname in sort(collect(keys(results)))
            @printf(
                io_handle,
                "%40.15f%40.15f",
                real(results[opname][end]),
                imag(results[opname][end])
            )
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
            # TODO Use built-in trace function, do not create an MPS from scratch each time!
            @printf(io_handle, "%40.15f", real(inner(MPS(kwargs[:sites], "vecId"), psi)))
        else
            @printf(io_handle, "%40.15f", norm(psi))
        end
        @printf(io_handle, "\n")
        flush(io_handle)
    end

    return nothing
end

function printoutput_data(io_handle, cb, state1::MPS, state2::MPS; kwargs...)
    if !isnothing(io_handle)
        results = measurements(cb)
        @printf(io_handle, "%40.15f", measurement_ts(cb)[end])
        for opname in sort(collect(keys(results)))
            @printf(
                io_handle,
                "%40.15f%40.15f",
                real(results[opname][end]),
                imag(results[opname][end])
            )
        end

        ol = dot(state1, state2)
        @printf(io_handle, "%40.15f%40.15f", real(ol), imag(ol))

        if get(kwargs, :store_psi0, false)
            states0 = get(kwargs, :psi0, nothing)
            overlap = [dot(states0[1], state1), dot(states0[2], state2)]
            for ol in overlap
                @printf(io_handle, "%40.15f%40.15f", real(ol), imag(ol))
            end
        end

        # Print the norm of the trace of the state, depending on whether the MPS represents
        # a pure state or a vectorized density matrix.
        isvectorized = get(kwargs, :vectorized, false)
        if isvectorized
            # TODO Use built-in trace function, do not create an MPS from scratch each time!
            @printf(io_handle, "%40.15f", real(inner(MPS(kwargs[:sites], "vecId"), state1)))
            @printf(io_handle, "%40.15f", real(inner(MPS(kwargs[:sites], "vecId"), state2)))
        else
            @printf(io_handle, "%40.15f", norm(state1))
            @printf(io_handle, "%40.15f", norm(state2))
        end
        @printf(io_handle, "\n")
        flush(io_handle)
    end

    return nothing
end

function printoutput_ranks(ranks_handle, cb, states::MPS...)
    if !isnothing(ranks_handle)
        @printf(ranks_handle, "%40.15f", measurement_ts(cb)[end])

        for state in states
            for bonddim in ITensors.linkdims(state)
                @printf(ranks_handle, "%10d", bonddim)
            end
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

function append_if_not_null(filename::AbstractString, str::AbstractString)
    if filename != "/dev/null"
        return filename * str
    else
        return filename
    end
end
