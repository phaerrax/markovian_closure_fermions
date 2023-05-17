export TEvoCallback,
    NoTEvoCallback,
    LocalMeasurementCallback,
    SpecCallback,
    opPos,
    createObs,
    LocalPosMeasurementCallback,
    LocalPosVecMeasurementCallback,
    measurement_ts

"""
A TEvoCallback can implement the following methods:

- apply!(cb::TEvoCallback, psi ; t, kwargs...): apply the callback with the
current state `psi` (e.g. perform some measurement)

- checkdone!(cb::TEvoCallback, psi; t, kwargs...): check whether some criterion
to stop the time evolution is satisfied (e.g. convergence of cbervable, error
too large) and return `true` if so.

- callback_dt(cb::TEvoCallback): time-steps at which the callback needs access
for the wave-function (e.g. for measurements). This is used for TEBD evolution
where several time-steps are bunched together to reduce the cost.
"""
abstract type TEvoCallback end

"""
    NoTEvoCallback is a trivial implementation of an evolution callback (<:TEvoCallback)
    object.
"""
struct NoTEvoCallback <: TEvoCallback end

apply!(cb::NoTEvoCallback, args...; kwargs...) = nothing
checkdone!(cb::NoTEvoCallback, args...; kwargs...) = false
callback_dt(cb::NoTEvoCallback) = 0

"""
    A Measurement object is an alias for `Vector{Vector{Float64}}`, in other words an
    array of arrays of real numbers.

    Given a Measurement `M`, the result for the measurement at step `n` and site `i` is
    `M[n][i]`.
"""
const Measurement = Vector{Vector{Float64}}

struct LocalMeasurementCallback <: TEvoCallback
    "An array of operators that must be measured at each time step"
    ops::Vector{String}
    "The basis of sites used to define the MPS and MPO for the calculations"
    sites::Vector{<:Index}
    "A dictionary containing the measured values of the observables at each time step"
    measurements::Dict{String,Measurement}
    "The array of times"
    ts::Vector{Float64}
    # Measurement time-step
    dt_measure::Float64
end

"""
    LocalMeasurementCallback(ops::Vector{String},
                             sites::Vector{<:Index},
                             dt_measure::Float64)

Construct a LocalMeasurementCallback, providing an array `ops` of operator names which are
strings recognized by the `op` function. Each of these operators will be measured on every
site during every step of the time evolution, and the results recorded inside the object as
a Measurement for later analysis. The array `sites` is the basis of sites used to define
the MPS and MPO for the calculations.
"""
function LocalMeasurementCallback(
    ops::Vector{String},
    sites::Vector{<:Index},
    dt_measure::Float64,
)
    return LocalMeasurementCallback(
        ops,
        sites,
        Dict(o => Measurement[] for o in ops),
        Vector{Float64}(),
        dt_measure,
    )
end

measurement_ts(cb::LocalMeasurementCallback) = cb.ts
measurements(cb::LocalMeasurementCallback) = cb.measurements
callback_dt(cb::LocalMeasurementCallback) = cb.dt_measure
ops(cb::LocalMeasurementCallback) = cb.ops
sites(cb::LocalMeasurementCallback) = cb.sites

"""
    opPos(op::String, pos::Integer)

An opPos object is an operator `op` (a string recognised by ITensors' `op` function)
attached to a specific site `pos`.
"""
struct opPos
    op::String
    pos::Integer
end

struct LocalPosMeasurementCallback <: TEvoCallback
    ops::Vector{opPos}
    sites::Vector{<:Index}
    measurements::Dict{String,Measurement}
    ts::Vector{Float64}
    dt_measure::Float64
end

function createObs(lookat)
    vobs = opPos[]
    for a in lookat
        push!(vobs, opPos(a[1], a[2]))
    end
    return vobs
end

"""
    LocalPosMeasurementCallback(ops::Vector{opPos},
                                sites::Vector{<:Index},
                                dt_measure::Float64)

Construct a LocalPosMeasurementCallback, providing an array `ops` of opPos objects which
represent operators associated to specific sites. Each of these operators will be measured
on the given site during every step of the time evolution, and the results recorded inside
the LocalPosMeasurementCallback object as a Measurement for later analysis. The array
`sites` is the basis of sites used to define the MPS and MPO for the calculations.
"""
function LocalPosMeasurementCallback(
    ops::Vector{opPos},
    sites::Vector{<:Index},
    dt_measure::Float64,
)
    return LocalPosMeasurementCallback(
        ops,
        sites,
        Dict(o.op * "_" * string(o.pos) => Measurement[] for o in ops),
        Vector{Float64}(),
        dt_measure,
    )
end

# These functions replicate the behaviour of LocalMeasurementCallback above.
measurement_ts(cb::LocalPosMeasurementCallback) = cb.ts
measurements(cb::LocalPosMeasurementCallback) = cb.measurements
callback_dt(cb::LocalPosMeasurementCallback) = cb.dt_measure
ops(cb::LocalPosMeasurementCallback) = cb.ops
sites(cb::LocalPosMeasurementCallback) = cb.sites

struct LocalPosVecMeasurementCallback <: TEvoCallback
    ops::Vector{opPos}
    sites::Vector{<:Index}
    measurements::Dict{String,Measurement}
    ts::Vector{Float64}
    dt_measure::Float64
end

"""
    LocalPosVecMeasurementCallback(
        ops::Vector{opPos}, sites::Vector{<:Index}, dt_measure::Float64
    )

Like LocalPosMeasurementCallback, but with vectorized operators.
"""
function LocalPosVecMeasurementCallback(
    ops::Vector{opPos}, sites::Vector{<:Index}, dt_measure::Float64
)
    return LocalPosVecMeasurementCallback(
        ops,
        sites,
        Dict(o.op * "_" * string(o.pos) => Measurement[] for o in ops),
        Vector{Float64}(),
        dt_measure,
    )
end

# These functions replicate the behaviour of LocalMeasurementCallback above.
measurement_ts(cb::LocalPosVecMeasurementCallback) = cb.ts
measurements(cb::LocalPosVecMeasurementCallback) = cb.measurements
callback_dt(cb::LocalPosVecMeasurementCallback) = cb.dt_measure
ops(cb::LocalPosVecMeasurementCallback) = cb.ops
sites(cb::LocalPosVecMeasurementCallback) = cb.sites

function Base.show(io::IO, cb::Union{LocalMeasurementCallback,LocalPosMeasurementCallback})
    println(io, "LocalMeasurementCallback")
    # Print the list of operators
    println(io, "Operators: ", ops(cb))
    if length(measurement_ts(cb)) > 0
        println(
            io,
            "Measured times: ",
            callback_dt(cb):callback_dt(cb):measurement_ts(cb)[end],
        )
    else
        println(io, "No measurements performed")
    end
end

# function Base.show(io::IO, cb::LocalPosMeasurementCallback)
#     println(io, "LocalMeasurementCallback")
#     println(io, "Operators: ", ops(cb))
#     if length(measurement_ts(cb))>0
#         println(io, "Measured times: ", callback_dt(cb):callback_dt(cb):measurement_ts(cb)[end])
#     else
#         println(io, "No measurements performed")
#     end
# end

"""
    function measure_localops!(cb::LocalMeasurementCallback, wf::ITensor, i::Int)

Measure each operator defined inside the callback object `cb` on the tensor `wf`, which
represents the sit `i` of a bigger MPS.

This requires that the MPS where `wf` comes from has been properly orthogonalized such
that its orthogonality center is right on site `i`. This is understood when calculating
expectation values using this function; otherwise, it gives unexpected results.
"""
function measure_localops!(cb::LocalMeasurementCallback, wf::ITensor, n::Int)
    # Loop over every operator contained in `cb`.
    for opname in ops(cb)
        # op(sites(cb), opname, n) returns an ITensors operator named opname on the n-th
        # index in the array sites(cb).
        m = dot(wf, noprime(op(sites(cb), opname, n) * wf))
        imag(m) > 1e-8 && (@warn "Non-zero imaginary part when measuring $opname")
        measurements(cb)[opname][end][n] = real(m)
        # measurements(cb)[opname][end] is the last line in the measurements of opname,
        # which we (must) have created in apply! before calling this function.
    end
end

# This function needs to be modified when measuring a vectorized mixed state.
# We need now the whole MPS to be passed to the function, since (unless we are
# able to provide a proof to the contrary), we need the full MPS to compute the
# element of the state tensor we need: if we want to measure the expectation value of
# an operator aⱼ on site j, the remaining sites of the MPS do not contract to give
# the identity as in the pure-state scenario, but they must be contracted with vecId.

"""
    isoncurrentbond(op::opPos, i::Int, alg)

Return whether the local operator `op` is associated to bond `i` (or bonds `i` and `i+1`,
if the algorithm `alg` is 2-site TDVP).
"""
function isoncurrentbond(op::opPos, bond::Int, alg)
    return (
        (alg isa TDVP2 && (op.pos == bond + 1 || (op.pos == 1 && bond == 1))) ||
        (alg isa TDVP1 && op.pos == bond)
    )
end

"""
    measure_localops!(
        cb::LocalPosMeasurementCallback,
        wf::ITensor,
        i::Int,
        alg
    )

Measure each operator defined inside the callback object `cb` on the given tensor `wf`
at the selected bond or site `i` (depending on the algorithm `alg`).
"""
function measure_localops!(
    cb::LocalPosMeasurementCallback,
    wf::ITensor,
    bond::Int,
    alg,
)
    operators_thisbond = filter(op -> isoncurrentbond(op, bond, alg), ops(cb))

    if !isempty(operators_thisbond)
        for o in operators_thisbond
            # We replace the placeholder "Norm" so that we compute the correct operator.
            if o.op == "Norm"
                m = norm(wf)
            else
                m = dot(wf, noprime(op(sites(cb), o.op, bond) * wf))
            end

            imag(m) > 1e-5 && (@warn "encountered finite imaginary part when measuring $o")

            # NOTE Since we don't have an operator for each site, we have a single value
            # for each operator in cb, and operators associated to different sites have
            # different entries in the dictionary.
            # Brought to an extreme, when using the other measure_localops! method a single
            # operator A on every site of the chain would have a single dicionary entry,
            # a list whose elements are the expectation values of A on each site.
            # With this method, instead, if we have different operators Aᵢ for each
            # site, they are not bunched up in a single line, but there will be a
            # different dictionary entry for each one of them.
            measurements(cb)[o.op * "_" * string(o.pos)][end][1] = real(m)
        end
    end

    return nothing
end

"""
    measure_localops!(cb::LocalPosMeasurementCallback, psi::MPS, i::Int, alg)

Measure each operator defined inside the callback object `cb` on the given MPS `psi`
at the selected site `i` (depending on the algorithm `alg`).

The function is implemented so as to contract the minimum number of tensor sites as
needed to provide a correct result, given `i` and the orthocentre of the MPS.
"""
function measure_localops!(cb::LocalPosMeasurementCallback, psi::MPS, site::Int, alg)
    # Get the operators defined on the given site.
    operators_thisbond = filter(op -> isoncurrentbond(op, bond, alg), ops(cb))

    # Why do we need this function?
    # We are sweeping right-to-left during a TDVP step, and we have just completed the
    # evolution on site `site`.
    # The MPS `psi` is now such that `psi[site]` is correctly updated at time `t`, but
    # the orthocentre is on site `psi[site - 1]` (the step involves decomposing `psi[site]`
    # into `Q * R` then leaving `Q` at the site and incorporating `R` into `psi[site - 1]`).
    # This means that we cannot just use `psi[site]` to calculate the expectation value
    # of an operator at site `site`, nor can we use the orthocentre `psi[site - 1]` for
    # operators at site `site - 1` since it has not yet completed its evolution step.
    # So what do we do?
    # We take all sites between `site` and the orthocentre of the MPS, form an ITensor
    # with those, and sandwich the operator.
    oc = ortho_lims(psi) # This is a range
    if site in oc
        site_range = oc
    elseif site < first(oc)
        site_range = site:last(oc)
    else # site > last(oc)
        site_range = first(oc):site
    end

    for o in operators_thisbond
        if o.op == "Norm"
            x = norm(wf)
        else
            x = ITensor.OneITensor()
            for s in site_range
                if s == o.pos
                    x *= dot(psi[s], noprime(op(sites(cb), o.op, s) * psi[s]))
                else
                    x *= dot(psi[s], psi[s])
                end
            end
            x = scalar(x)
        end
        imag(x) > 1e-5 && (@warn "encountered finite imaginary part when measuring $o")

        # NOTE Since we don't have an operator for each site, we have a single value
        # for each operator in cb, and operators associated to different sites have
        # different entries in the dictionary.
        # Brought to an extreme, when using the other measure_localops! method a single
        # operator A on every site of the chain would have a single dicionary entry,
        # a list whose elements are the expectation values of A on each site.
        # With this method, instead, if we have different operators Aᵢ for each
        # site, they are not bunched up in a single line, but there will be a
        # different dictionary entry for each one of them.
        measurements(cb)[o.op * "_" * string(o.pos)][end][1] = real(x)
    end

    return nothing
end

"""
    measure_localops!(
        cb::LocalPosVecMeasurementCallback,
        ψ::MPS,
        i::Int,
        alg
    )

Measure each operator defined inside the callback object `cb` on the MPS `ψ` at the
selected bond or site `i` (depending on the algorithm `alg`).

Use this version with vectorized operators.
"""
function measure_localops!(
    cb::LocalPosVecMeasurementCallback,
    ψ::MPS,
    bond::Int,
    alg,
)
    operators_thisbond = filter(op -> isoncurrentbond(op, bond, alg), ops(cb))

    if !isempty(operators_thisbond)
        for o in operators_thisbond
            V = ITensor(1.0)
            # The norm (aka the trace) is computed contracting with vecId on every site,
            # so we replace the placeholder "Norm" so that we have the correct operator.
            if o.op == "Norm"
                opnames = repeat(["vecId"], length(ψ))
            else
                opnames = [j == o.pos ? o.op : "vecId" for j in 1:length(ψ)]
            end

            for j in 1:length(ψ)
                # Contract with the operator on the site associated to it, with vecId
                # on the other sites.
                V *= ψ[j] * state(opnames[j], siteind(ψ, j))
            end
            m = scalar(V)

            imag(m) > 1e-5 && (@warn "encountered finite imaginary part when measuring $o")

            # NOTE Since we don't have an operator for each site, we have a single value
            # for each operator in cb, and operators associated to different sites have
            # different entries in the dictionary.
            # Brought to an extreme, when using the other measure_localops! method a single
            # operator A on every site of the chain would have a single dicionary entry,
            # a list whose elements are the expectation values of A on each site.
            # With this method, instead, if we have different operators Aᵢ for each
            # site, they are not bunched up in a single line, but there will be a
            # different dictionary entry for each one of them.
            measurements(cb)[o.op * "_" * string(o.pos)][end][1] = real(m)
        end
    end

    return nothing
end

"""
    apply!(cb::LocalMeasurementCallback, psi; t, sweepend, sweepdir, bond, alg, kwargs...)

Calculates the expectation values of the operators stored in `cb` on `state`, if the
conditions appropriate to the evolution algorithm are met.
"""
function apply!(
    cb::LocalMeasurementCallback,
    psi;
    t,
    sweepend,
    sweepdir,
    bond,
    alg,
    kwargs...,
)
    if !isempty(measurement_ts(cb))
        # If there's already one element in the measurement_ts list, then this is not
        # the first step, and we set prev_t to the time instant of the previous step.
        prev_t = measurement_ts(cb)[end]
    else
        # Otherwise, this is the beginning of the time evolution, therefore the previous
        # time instant is zero.
        prev_t = 0
    end

    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) &&
       sweepend &&
       (bond % 2 == 1 || !(alg isa TEBDalg))
        # If the following hold:
        # 1) t-prev_t = callback_dt(cb) or t-prev_t = 0
        # 2) we are at the end of a sweep
        # 3) if we are using TEBD, then the bond is an odd bond
        #    then we can perform the measurement.
        #
        # For TEBD algorithms we want to perform measurements only in the final
        # sweep over odd bonds. For TDVP we can perform measurements to the right of
        # each bond when sweeping back left.
        # We perform measurements only at the end of a sweep (TEBD: for finishing
        # evolution over the measurement time interval; TDVP: when sweeping left)
        # and at measurement steps.
        if t != prev_t # then it must be that t - prev_t ≈ callback_dt(cb)
            push!(measurement_ts(cb), t)
            foreach(x -> push!(x, zeros(length(psi))), values(measurements(cb)))
            # This means: push [0,…,0] to v, ∀ v ∈ values(measurements(cb)).
            # The function values returns an iterator over all values in a dictionary.
            # This is necessary because measure_localops! modifies the values in the
            # dictionary, it doesn't push them from scratch; in other words, the new
            # line which containes the new measurement must already be there when
            # measure_localops! is called (this is what we're doing here).
        end
        # We pass the relevant part of the state MPS to measure_localops!, so that we
        # can retrieve the measurements relative to the currently updated blocks.
        if alg isa TDVP1
            # TDVP1 updates blocks one by one, so each site of the MPS is traversed
            # during the right-to-left sweep.
            wf = psi[bond]
            measure_localops!(cb, wf, bond)
        elseif (alg isa TDVP2 && bond == length(sites(cb))-1)
            # The first step in the right-to-left sweep involves the (N-1, N) bond,
            # then (N-2, N-1), until (1, 2).
            # If bond is the first index of the pair, then we need to treat the first
            # step of the right-to-left sweep explicitly, so that we actually measure
            # the observables on the last site too.
            wf = psi[bond] * psi[bond+1]
            measure_localops!(cb, wf, bond + 1)
            measure_localops!(cb, wf, bond)
        elseif alg isa TDVP2
            # Now we surely have bond != length(sites(cb))-1)
            wf = psi[bond] * psi[bond+1]
            measure_localops!(cb, wf, bond)
        elseif alg isa TEBDalg
            wf = psi[bond] * psi[bond+1]
            measure_localops!(cb, wf, bond)
        elseif bond == 1 # ???
            measure_localops!(cb, wf, bond)
        end
    end
end

"""
    apply!(
        cb::LocalPosMeasurementCallback, state; t, sweepend, sweepdir, bond, alg, kwargs...
    )

Calculates the expectation values of the operators stored in `cb` on `state`, if the
conditions appropriate to the evolution algorithm are met.
"""
function apply!(
    cb::LocalPosMeasurementCallback,
    state;
    t,
    sweepend,
    sweepdir,
    bond,
    alg,
    kwargs...,
)
    prev_t = !isempty(measurement_ts(cb)) ? measurement_ts(cb)[end] : 0

    # Perform measurements only at the end of a sweep (TEBD: for finishing
    # evolution over the measurement time interval; TDVP: when sweeping left)
    # and at measurement steps.
    # For TEBD algorithms we want to perform measurements only in the final
    # sweep over odd bonds. For TDVP we can perform measurements to the right of
    # each bond when sweeping back left.

    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) &&
       sweepend &&
       (bond % 2 == 1 || !(alg isa TEBDalg))
        if (t != prev_t || t == 0)
            push!(measurement_ts(cb), t)
            foreach(x -> push!(x, zeros(1)), values(measurements(cb)))
        end

        if alg isa TDVP2
            wf = state[bond] * state[bond+1]
            measure_localops!(cb, wf, bond + 1, alg)
        elseif alg isa TDVP1
            wf = state[bond]
            measure_localops!(cb, wf, bond, alg)
        elseif alg isa TEBDalg
            measure_localops!(cb, wf, bond, alg)
            # NOTE What should wf be here?
        end

        #I modified the condition above, so that if
        #the measurement is on the first site and bond ==1
        #what follows is unnecessary.

        # elseif bond==1
        #     #Specialize for first site
        #     pippo=opPos[]
        #     for el in ops(cb)
        #         if (el.pos == 1)
        #             push!(pippo,el)
        #         end
        #     end
        #     if(length(pippo)>0)
        #         wf = state[bond]*state[bond+1]
        #         measure_localops!(cb,pippo,wf,bond)
        #     end
        # end
    end

    return nothing
end

"""
    apply!(
        cb::LocalPosVecMeasurementCallback, state;
        t, sweepend, sweepdir, bond, alg, kwargs...
    )

Calculates the expectation values of the operators stored in `cb` on `state`, if the
conditions appropriate to the evolution algorithm are met.
"""
function apply!(
    cb::LocalPosVecMeasurementCallback,
    state;
    t,
    sweepend,
    sweepdir,
    bond,
    alg,
    kwargs...,
)
    prev_t = !isempty(measurement_ts(cb)) ? measurement_ts(cb)[end] : 0

    # Perform measurements only at the end of a sweep (TEBD: for finishing
    # evolution over the measurement time interval; TDVP: when sweeping left)
    # and at measurement steps.
    # For TEBD algorithms we want to perform measurements only in the final
    # sweep over odd bonds. For TDVP we can perform measurements to the right of
    # each bond when sweeping back left.

    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) &&
       sweepend &&
       (bond % 2 == 1 || !(alg isa TEBDalg))
        if (t != prev_t || t == 0)
            push!(measurement_ts(cb), t)
            foreach(x -> push!(x, zeros(1)), values(measurements(cb)))
        end

        if alg isa TDVP2
            measure_localops!(cb, state, bond + 1, alg)
        elseif alg isa TDVP1
            measure_localops!(cb, state, bond, alg)
        elseif alg isa TEBDalg
            measure_localops!(cb, state, bond, alg)
        end

        #I modified the condition above, so that if
        #the measurement is on the first site and bond ==1
        #what follows is unnecessary.

        # elseif bond==1
        #     #Specialize for first site
        #     pippo=opPos[]
        #     for el in ops(cb)
        #         if (el.pos == 1)
        #             push!(pippo,el)
        #         end
        #     end
        #     if(length(pippo)>0)
        #         wf = state[bond]*state[bond+1]
        #         measure_localops!(cb,pippo,wf,bond)
        #     end
        # end
    end

    return nothing
end

checkdone!(cb::LocalMeasurementCallback, args...) = false
checkdone!(cb::LocalPosMeasurementCallback, args...) = false
checkdone!(cb::LocalPosVecMeasurementCallback, args...) = false

struct SpecCallback <: TEvoCallback
    truncerrs::Vector{Float64}
    current_truncerr::Base.RefValue{Float64}
    entropies::Measurement
    bonddims::Vector{Vector{Int64}}
    bonds::Vector{Int64}
    ts::Vector{Float64}
    dt_measure::Float64
end

function SpecCallback(dt, psi::MPS, bonds::Vector{Int64} = collect(1:(length(psi)-1)))
    bonds = sort(unique(bonds))
    if maximum(bonds) > length(psi) - 1 || minimum(bonds) < 1
        throw("bonds must be between 1 and $(length(psi)-1)")
    end
    return SpecCallback(
        Vector{Float64}(),
        Ref(0.0),
        Measurement(),
        Vector{Vector{Int64}}(),
        bonds,
        Vector{Float64}(),
        dt,
    )
end

measurement_ts(cb::SpecCallback) = cb.ts

function ITensors.measurements(cb::SpecCallback)
    return Dict(
        "entropy" => cb.entropies,
        "bonddim" => cb.bonddims,
        "truncerrs" => cb.truncerrs,
    )
end

callback_dt(cb::SpecCallback) = cb.dt_measure
bonds(cb::SpecCallback) = cb.bonds

function Base.show(io::IO, cb::SpecCallback)
    println(io, "SpecCallback")
    if length(measurement_ts(cb)) > 0
        println(
            io,
            "Measured times: ",
            callback_dt(cb):callback_dt(cb):measurement_ts(cb)[end],
        )
    else
        println(io, "No measurements performed")
    end
end

function apply!(cb::SpecCallback, psi; t, sweepend, bond, spec, sweepdir, kwargs...)
    cb.current_truncerr[] += truncerror(spec)
    prev_t = length(measurement_ts(cb)) > 0 ? measurement_ts(cb)[end] : 0
    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend
        if t != prev_t
            push!(measurement_ts(cb), t)
            push!(cb.bonddims, zeros(Int64, length(cb.bonds)))
            push!(cb.entropies, zeros(length(cb.bonds)))
        end

        if bond in bonds(cb)
            i = findfirst(x -> x == bond, bonds(cb))
            cb.bonddims[end][i] = length(eigs(spec))
            cb.entropies[end][i] = entropy(spec)
        end
        if sweepdir == "right" && bond == length(psi) - 1
            push!(cb.truncerrs, cb.current_truncerr[])
        elseif sweepdir == "left" && bond == 1
            push!(cb.truncerrs, cb.current_truncerr[])
        end
    end
end

checkdone!(cb::SpecCallback, args...) = false
