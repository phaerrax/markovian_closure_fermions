export SuperfermionCallback

const SuperfermionSeries = Vector{ComplexF64}

struct SuperfermionCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,SuperfermionSeries}
    times::Vector{Float64}
    measure_timestep::Float64
end

"""
    SuperfermionCallback(ops::Vector{LocalOperator},
                          sites::Vector{<:Index},
                          measure_timestep::Float64)

Construct a SuperfermionCallback, providing an array `ops` of `LocalOperator` objects which
represent operators associated to specific sites. Each of these operators will be measured
on the given site during every step of the time evolution, and the results recorded inside
the SuperfermionCallback object as a SuperfermionSeries for later analysis. The array
`sites` is the basis of sites used to define the MPS and MPO for the calculations.
"""
function SuperfermionCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return SuperfermionCallback(
        operators,
        sites,
        Dict(op => SuperfermionSeries() for op in operators),
        # A single SuperfermionSeries for each operator in the list.
        Vector{Float64}(),
        measure_timestep,
    )
end

measurement_ts(cb::SuperfermionCallback) = cb.times
measurements(cb::SuperfermionCallback) = cb.measurements
callback_dt(cb::SuperfermionCallback) = cb.measure_timestep
ops(cb::SuperfermionCallback) = cb.operators
sites(cb::SuperfermionCallback) = cb.sites

function Base.show(io::IO, cb::SuperfermionCallback)
    println(io, "SuperfermionCallback")
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

checkdone!(cb::SuperfermionCallback, args...) = false

"""
    measure_localops!(cb::SuperfermionCallback, state::MPS, site::Int, alg::TDVP1)

Measure each operator defined inside the callback object `cb` on the state `state`
"""
function measure_localops!(cb::SuperfermionCallback, state::MPS, site::Int, alg::TDVP1)
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

function identity_sf(::SiteType"Fermion", sites)
    N = length(sites)
    @assert iseven(N)
    pairs = [
        add(MPS(sites[n:(n + 1)], "Emp"), MPS(sites[n:(n + 1)], "Occ"); alg="directsum") for
        n in 1:2:N if n + 1 ≤ N
    ]
    # Use the "direct sum" method for summing the MPSs so that ITensors doesn't waste time
    # re-orthogonalizing the result, which might also reverse the direction of some QN
    # arrows, and we don't want that.

    id = MPS(
        collect(Iterators.flatten((pairs[n][1], pairs[n][2]) for n in eachindex(pairs)))
    )

    vacuum = MPS(sites, "Emp")  # copy link indices from here
    # Here "Up" or "Dn" makes no difference, because we are interested only in the link
    # indices between sites 2j and 2j+1, and an even number of "Up" or "Dn" states doesn't
    # change the parity.

    for n in 2:2:(N - 1)
        id[n] *= state(linkind(vacuum, n), 1)
        id[n + 1] *= state(dag(linkind(vacuum, n)), 1)
    end
    return id
end

"""
    adj(x)

Returns the adjoint (conjugate transpose) of x. It can be an ITensor, an MPS or an MPO.
Note that it is not the same as ITensors.adjoint.
"""
adj(x) = swapprime(dag(x), 0 => 1)

function _sf_observable_mps(op, sites)
    id = identity_sf(SiteType("Fermion"), sites)
    x = mpo(sites, op)
    return apply(adj(x), id)
end

"""
    measure_localops!(cb::SuperfermionCallback, state::MPS, site::Int, alg::TDVP1vec)

Measure each operator defined inside the callback object `cb` on the state `state` at site
`site`.
"""
function measure_localops!(cb::SuperfermionCallback, state::MPS, site::Int, alg::TDVP1vec)
    # With TDVP1vec algorithms the situation is much simpler than with simple TDVP1: since
    # we need to contract any site which is not "occupied" (by the operator which is to be
    # measured) anyway with vec(I), we don't need to care about the orthocenter, we just
    # measure everything at the end of the sweep.

    for localop in ops(cb)
        # Strategy:
        # 1. create the identity MPS (the so-called “left vacuum”)
        # 2. apply the adjoint of the local operator to it (this way we can memoize it)
        # 3. contract the result with the state
        measurements(cb)[localop][end] = dot(_sf_observable_mps(localop, sites(cb)), state)
        # measurements(cb)[localop][end] is the last element in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end
end

function apply!(
    cb::SuperfermionCallback, state::MPS; t, sweepend, sweepdir, site, alg, kwargs...
)
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
