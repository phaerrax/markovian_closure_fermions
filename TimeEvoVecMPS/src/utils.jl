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

        columnheaders = ["time"]

        res = measurements(cb)
        for op in sort(collect(keys(res)))
            push!(columnheaders, name(op) * "_re", name(op) * "_im")
        end

        if get(kwargs, :store_psi0, false)
            push!(columnheaders, "overlap_re", "overlap_im")
        end

        push!(columnheaders, "Norm")

        println(io_handle, join(columnheaders, ","))
    end

    return io_handle
end

function writeheaders_data_double(io_file, cb; kwargs...)
    io_handle = nothing
    if !isnothing(io_file)
        io_handle = open(io_file, "w")

        columnheaders = ["time"]

        res = measurements(cb)
        for op in sort(collect(keys(res)))
            push!(columnheaders, name(op) * "_re", name(op) * "_im")
        end

        push!(columnheaders, "overlap12_re", "overlap12_im")
        if get(kwargs, :store_psi0, false)
            push!(
                columnheaders,
                "overlap1init_re",
                "overlap1init_im",
                "overlap2init_re",
                "overlap2init_im",
            )
        end

        push!(columnheaders, "Norm1", "Norm2")

        println(io_handle, join(columnheaders, ","))
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

        columnheaders = ["time"]

        for N in Ns
            append!(columnheaders, string.(1:(N - 1)))
        end

        println(ranks_handle, join(columnheaders, ","))
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
        println(times_handle, "walltime/s")
    end

    return times_handle
end

function printoutput_data(io_handle, cb, psi::MPS; kwargs...)
    if !isnothing(io_handle)
        results = measurements(cb)
        data = [measurement_ts(cb)[end]]
        for opname in sort(collect(keys(results)))
            x = results[opname][end]
            push!(data, real(x), imag(x))
        end

        if get(kwargs, :store_psi0, false)
            psi0 = get(kwargs, :psi0, nothing)
            overlap = dot(psi0, psi)  # FIXME What if psi0 is nothing?
            push!(data, real(overlap), imag(overlap))
        end

        # Print the norm of the trace of the state, depending on whether the MPS represents
        # a pure state or a vectorized density matrix.
        isvectorized = get(kwargs, :vectorized, false)
        if isvectorized
            # TODO Use built-in trace function, do not create an MPS from scratch each time!
            push!(data, real(inner(MPS(kwargs[:sites], "vecId"), psi)))
        else
            push!(data, norm(psi))
        end

        println(io_handle, join(data, ","))
        flush(io_handle)
    end

    return nothing
end

function printoutput_data(io_handle, cb, state1::MPS, state2::MPS; kwargs...)
    if !isnothing(io_handle)
        results = measurements(cb)
        data = [measurement_ts(cb)[end]]
        for opname in sort(collect(keys(results)))
            x = results[opname][end]
            push!(data, real(x), imag(x))
        end

        ol = dot(state1, state2)
        push!(data, real(ol), imag(ol))

        if get(kwargs, :store_psi0, false)
            states0 = get(kwargs, :psi0, nothing)
            overlap = [dot(states0[1], state1), dot(states0[2], state2)]
            for ol in overlap
                push!(data, real(ol), imag(ol))
            end
        end

        # Print the norm of the trace of the state, depending on whether the MPS represents
        # a pure state or a vectorized density matrix.
        isvectorized = get(kwargs, :vectorized, false)
        if isvectorized
            # TODO Use built-in trace function, do not create an MPS from scratch each time!
            vecId = MPS(kwargs[:sites], "vecId")
            push!(data, real(inner(vecId, state1)), real(inner(vecId, state2)))
        else
            push!(data, norm(state1), norm(state2))
        end

        println(io_handle, join(data, ","))
        flush(io_handle)
    end

    return nothing
end

function printoutput_ranks(ranks_handle, cb, states::MPS...)
    if !isnothing(ranks_handle)
        data = [measurement_ts(cb)[end]]

        for state in states
            push!(data, ITensors.linkdims(state)...)
        end

        println(ranks_handle, join(data, ","))
        flush(ranks_handle)
    end

    return nothing
end

function printoutput_stime(times_handle, stime::Real)
    if !isnothing(times_handle)
        println(times_handle, stime)
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
