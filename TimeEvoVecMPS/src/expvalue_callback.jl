export ExpValueCallback

const ExpValueSeries = Vector{ComplexF64}

struct ExpValueCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
    times::Vector{Float64}
    measure_timestep::Float64
end

"""
    ExpValueCallback(ops::Vector{LocalOperator},
                          sites::Vector{<:Index},
                          measure_timestep::Float64)

Construct a ExpValueCallback, providing an array `ops` of LocalOperator objects which
represent operators associated to specific sites. Each of these operators will be measured
on the given site during every step of the time evolution, and the results recorded inside
the ExpValueCallback object as a ExpValueSeries for later analysis. The array
`sites` is the basis of sites used to define the MPS and MPO for the calculations.
"""
function ExpValueCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return ExpValueCallback(
        operators,
        sites,
        Dict(op => ExpValueSeries() for op in operators),
        # A single ExpValueSeries for each operator in the list.
        Vector{Float64}(),
        measure_timestep,
    )
end

measurement_ts(cb::ExpValueCallback) = cb.times
measurements(cb::ExpValueCallback) = cb.measurements
callback_dt(cb::ExpValueCallback) = cb.measure_timestep
ops(cb::ExpValueCallback) = cb.operators
sites(cb::ExpValueCallback) = cb.sites

function Base.show(io::IO, cb::ExpValueCallback)
    println(io, "ExpValueCallback")
    # Print the list of operators
    println(io, "Operators: ", join(name.(ops(cb)), ", ", " and "))
    if Base.length(measurement_ts(cb)) > 0
        println(
            io, "Measured times: ", callback_dt(cb):callback_dt(cb):measurement_ts(cb)[end]
        )
    else
        println(io, "No measurements performed")
    end
end

checkdone!(cb::ExpValueCallback, args...) = false

"""
    measure_localops!(cb::ExpValueCallback, ψ::MPS, site::Int, alg::TDVP1)

Measure each operator defined inside the callback object `cb` on the state `ψ` at site `i`.
"""
function measure_localops!(cb::ExpValueCallback, ψ::MPS, site::Int, alg::TDVP1)
    # Since the operators may be defined on more than one site, we need to check that
    # all the sites in their domain have been completely updated: this means that we must
    # wait until the final sweep of the evolution step has passed each site in the domain.
    # Note that this function gets called (or should be called) only if the current sweep
    # is the final sweep in the step.

    # When `ψ[site]` has been updated during the leftwards sweep, the orthocentre lies on
    # site `max(1, site-1)`, and all `ψ[n]` with `n >= site` are correctly updated.
    measurable_operators = filter(op -> site <= first(domain(op)), ops(cb))
    s = siteinds(ψ)
    for localop in measurable_operators
        # This works, but calculating the MPO from scratch every time might take too much
        # time, especially when it has to be repeated thousands of times. For example,
        # executing TimeEvoVecMPS.mpo(s, o) with
        #   s = siteinds("Osc", 400; dim=16)
        #   o = LocalOperator(Dict(20 => "A", 19 => "Adag"))
        # takes 177.951 ms (2313338 allocations: 329.80 MiB).
        # Memoizing this function allows us to cut the time (after the first call, which is
        # expensive anyway since Julia needs to compile the function) to 45.368 ns
        # (1 allocation: 32 bytes) for each call.
        measurements(cb)[localop][end] = dot(ψ', mpo(s, localop), ψ)
        # measurements(cb)[localop][end] is the last line in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end
end

"""
    measure_localops!(cb::ExpValueCallback, ψ::MPS, site::Int, alg::TDVP1vec)

Measure each operator defined inside the callback object `cb` on the state `ψ` at site `i`.
"""
function measure_localops!(cb::ExpValueCallback, ψ::MPS, site::Int, alg::TDVP1vec)
    # With TDVP1vec algorithms the situation is much simpler than with simple TDVP1: since
    # we need to contract any site which is not "occupied" (by the operator which is to be
    # measured) anyway with vec(I), we don't need to care about the orthocenter, we just
    # measure everything at the end of the sweep.

    for localop in ops(cb)
        # Transform each `localop` into an MPS, filling with `vId` states.
        measurements(cb)[localop][end] = dot(mps(sites(cb), localop), ψ)
        # measurements(cb)[localop][end] is the last element in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end
end

function apply!(cb::ExpValueCallback, state; t, sweepend, sweepdir, site, alg, kwargs...)
    if isempty(measurement_ts(cb))
        prev_t = 0
    else
        prev_t = measurement_ts(cb)[end]
    end

    # We perform measurements only at the end of a sweep and at measurement steps.
    # For TDVP we can perform measurements to the right of each site when sweeping back left.
    if !(alg isa TDVP1 || alg isa TDVP1vec)
        error("apply! function only implemented for TDVP1 algorithms.")
    end

    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend
        if (t != prev_t || t == 0)
            # Add the current time to the list of time instants at which we measured
            # something.
            push!(measurement_ts(cb), t)
            # Create a new slot in which we will put the measurement result.
            foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
        end
        measure_localops!(cb, state, site, alg)
    end

    return nothing
end
