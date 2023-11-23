export LocalOperator, LocalOperatorCallback

"""
    LocalOperator(factors::Vector{AbstractString}, lind::Int)

A LocalOperator represents a product of local operators, whose names (strings, as recognized
by ITensors) are specified by `factors` and acting on consecutive sites, starting from
`lind`. For example, the operator ``I ⊗ A ⊗ B ⊗ C`` would be represented as
``julia
    LocalOperator(["A", "B", "C"], 2)
``
"""
struct LocalOperator
    factors::Vector{AbstractString}
    lind::Int
end

factors(op::LocalOperator) = op.factors
length(op::LocalOperator) = Base.length(op.factors)
domain(op::LocalOperator) = (op.lind):(op.lind + length(op) - 1)
name(op::LocalOperator) = *(["$f{$i}" for (i, f) in zip(domain(op), factors(op))]...)

"""
    onsite(op::LocalOperator, site::Int)

Return, if it exists, the factor in `op` which acts on site `site`.
"""
onsite(op::LocalOperator, site::Int) = op.factors[site - op.lind + 1]

struct LocalOperatorCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,Measurement}
    times::Vector{Float64}
    measure_timestep::Float64
end

"""
    LocalOperatorCallback(ops::Vector{LocalOperator},
                          sites::Vector{<:Index},
                          measure_timestep::Float64)

Construct a LocalOperatorCallback, providing an array `ops` of LocalOperator objects which
represent operators associated to specific sites. Each of these operators will be measured
on the given site during every step of the time evolution, and the results recorded inside
the LocalOperatorCallback object as a Measurement for later analysis. The array
`sites` is the basis of sites used to define the MPS and MPO for the calculations.
"""
function LocalOperatorCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return LocalOperatorCallback(
        operators,
        sites,
        Dict(op => Measurement() for op in operators),
        # A single Measurement for each operator in the list.
        Vector{Float64}(),
        measure_timestep,
    )
end

measurement_ts(cb::LocalOperatorCallback) = cb.times
measurements(cb::LocalOperatorCallback) = cb.measurements
callback_dt(cb::LocalOperatorCallback) = cb.measure_timestep
ops(cb::LocalOperatorCallback) = cb.operators
sites(cb::LocalOperatorCallback) = cb.sites

function Base.show(io::IO, cb::LocalOperatorCallback)
    println(io, "LocalOperatorCallback")
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

"""
    smart_contract(A::LocalOperator, ψ::MPS, sites)

Return the expectation value ``⟨ψ|A|ψ⟩`` contracting only sites in `A`'s domain.
It is assumed that `ψ`'s orthocentre lies within `sites`.
"""
function smart_contract(A::LocalOperator, ψ::MPS, sites)
    a = ITensors.OneITensor()
    v = ITensors.OneITensor()
    s = siteinds(ψ)
    for n in sites
        if n in domain(A)
            a *= op(onsite(A, n), s, n)
        else
            a *= delta(s[n], s[n]')
        end
        v *= ψ[n]
    end
    x = dag(prime(v; tags="Site")) * a * v
    return scalar(x)
end

"""
    measure_localops!(cb::LocalOperatorCallback, ψ::MPS, site::Int, alg::TDVP1)

Measure each operator defined inside the callback object `cb` on the state `ψ` at site `i`.
"""
function measure_localops!(cb::LocalOperatorCallback, ψ::MPS, site::Int, alg::TDVP1)
    # Since the operators may be defined on more than one site, we need to check that
    # all the sites in their domain have been completely updated: this means that we must
    # wait until the final sweep of the evolution step has passed each site in the domain.
    # Note that this function gets called (or should be called) only if the current sweep
    # is the final sweep in the step.

    # When `ψ[site]` has been updated during the leftwards sweep, the orthocentre lies on
    # site `max(1, site-1)`, and all `ψ[n]` with `n >= site` are correctly updated.
    measurable_operators = filter(op -> site <= first(domain(op)), ops(cb))
    for localop in measurable_operators
        # We need to transform each `localop` into an ITensors operator.
        localops = [
            ITensors.op(opname, sites(cb), i) for
            (i, opname) in zip(domain(localop), factors(localop))
        ]
        # Multiply all factors together, to create a single ITensor.
        op = *(localops...)
        oc = Tensors.orthocenter(ψ)
        if oc in domain(op)
            site_range = domain(op)
        elseif oc < first(domain(op))
            site_range = oc:last(domain(op))
        else  # oc > last(domain(op))
            site_range = first(domain(op)):oc
        end
        # `site_range` should be the smallest range containing both the state's orthocenter
        # and the domain of the operator.
        m = smart_contract(op, ψ, site_range)
        imag(m) > 1e-8 &&
            (@warn "Imaginary part when measuring $(name(localop)): $(imag(m))")
        measurements(cb)[localop][end] = real(m)
        # measurements(cb)[opname][end] is the last line in the measurements of opname,
        # which we (must) have created in apply! before calling this function.
    end
end

"""
    fill(sites::Vector{<:Index}, lop::LocalOperator)

Return an MPS with the factors in `lop` or `vId` if the site is not in the domain.
"""
function fill(sites::Vector{<:Index}, lop::LocalOperator)
    return MPS(ComplexF64, sites, [i in domain(lop) ? onsite(lop, i) : "vId" for i in 1:Base.length(sites)])
    # The MPS needs to be complex, in general, since
end

"""
    measure_localops!(cb::LocalOperatorCallback, ψ::MPS, site::Int, alg::TDVP1vec)

Measure each operator defined inside the callback object `cb` on the state `ψ` at site `i`.
"""
function measure_localops!(cb::LocalOperatorCallback, ψ::MPS, site::Int, alg::TDVP1vec)
    # With TDVP1vec algorithms the situation is much simpler than with simple TDVP1: since
    # we need to contract any site which is not "occupied" (by the operator which is to be
    # measured) anyway with vec(I), we don't need to care about the orthocenter, we just
    # measure everything at the end of the sweep.

    for localop in ops(cb)
        # Transform each `localop` into an MPS, filling with `vId` states.
        m = dot(fill(sites(cb), localop), ψ)
        imag(m) > 1e-8 &&
            (@warn "Imaginary part when measuring $(name(localop)): $(imag(m))")
        measurements(cb)[localop][end] = real(m)
        # measurements(cb)[opname][end] is the last element in the measurements of opname,
        # which we (must) have created in apply! before calling this function.
    end
end

function apply!(
    cb::LocalOperatorCallback, state; t, sweepend, sweepdir, site, alg, kwargs...
)
    prev_t = !isempty(measurement_ts(cb)) ? measurement_ts(cb)[end] : 0

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