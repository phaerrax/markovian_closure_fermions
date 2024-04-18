export jointtdvp1!

using ITensors: position!
using ITensors.ITensorMPS: set_nsite!

"""
    jointtdvp1!(solver, states::Tuple{MPS, MPS}, Hs::Vector{MPO}, dt::Number, tf::Number; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i Hⱼ ψₜ`` using the one-site TDVP algorithm,
for each state ``ψ`` in `states`, representing the state of the system, while the ``Hⱼ`` are
terms whose sum gives the Hamiltonian operator of the system.
Instead of returning the step-by-step expectation value of operators on a single state, it
returns quantities of the form ``⟨ψ₁,Aψ₂⟩`` for each operator ``A`` in the given callback
object, plus the identity.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `states::Tuple{MPS, MPS}`: a pair `(psi1, psi2)` of states to be evolved at the same time.
- `⃗Hs:Vector{MPO}`: a list of MPOs.
- `dt::Number`: time step of the evolution.
- `tf::Number`: end time of the evolution.
"""
function jointtdvp1!(
    solver,
    states::Tuple{MPS,MPS},
    Hs::Vector{MPO},
    time_step::Number,
    tf::Number;
    kwargs...,
)
    # (Copied from ITensorsTDVP)
    for H in Hs
        for psi in states
            ITensors.check_hascommoninds(siteinds, H, psi)
            ITensors.check_hascommoninds(siteinds, H, psi')
        end
    end
    Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return jointtdvp1!(solver, states, PHs, time_step, tf; kwargs...)
end

"""
    jointtdvp1!(solver, states::Tuple{MPS, MPS}, H::MPO, dt::Number, tf::Number; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the one-site TDVP algorithm,
for each state ``ψ`` in `states`, representing the state of the system, and `H` is the
Hamiltonian operator of the system.
Instead of returning the step-by-step expectation value of operators on a single state, it
returns quantities of the form ``⟨ψ₁,Aψ₂⟩`` for each operator ``A`` in the given callback
object, plus the identity.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `states::Tuple{MPS, MPS}`: a pair `(psi1, psi2)` of states to be evolved at the same time.
- `dt::Number`: time step of the evolution.
- `H::MPO`: the Hamiltonian operator.
- `tf::Number`: end time of the evolution.
"""
function jointtdvp1!(
    solver, states::Tuple{MPS,MPS}, H::MPO, timestep::Number, tf::Number; kwargs...
)
    return jointtdvp1!(solver, states, ProjMPO(H), timestep, tf; kwargs...)
end

"""
    jointtdvp1!(solver, states::Tuple{MPS,MPS},PH::ProjMPO, dt::Number, tf::Number; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the one-site TDVP algorithm,
for each state ``ψ`` in `states`, representing the state of the system, and `PH` is a
ProjMPO object encoding the Hamiltonian operator of the system.
Instead of returning the step-by-step expectation value of operators on a single state, it
returns quantities of the form ``⟨ψ₁,Aψ₂⟩`` for each operator ``A`` in the given callback
object, plus the identity.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `states::Tuple{MPS, MPS}`: a pair `(psi1, psi2)` of states to be evolved at the same time.
- `PH`: a ProjMPO-like operator encoding the Hamiltonian operator.
- `tf::Number`: end time of the evolution.
"""
function jointtdvp1!(
    solver, states::Tuple{MPS,MPS}, PH, timestep::Number, tf::Number; kwargs...
)
    nsteps = Int(tf / timestep)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)
    decomp = get(kwargs, :which_decomp, "qr")

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Usually TDVP is used for ordinary time evolution according to a Hamiltonian given
    # by `H`, and a real-valued time step `timestep`, combined in the evolution operator
    # U(-itH).
    # Passing an imaginary time step iτ (and `tf`) as an argument results in an evolution
    # according to the operator U(-τH), useful for thermalization processes.
    Δt = im * timestep
    imag(Δt) == 0 && (Δt = real(Δt)) # Discard imaginary part if time step is real.

    store_state0 && (states0 = copy.(states))

    @show io_file
    io_handle = writeheaders_data_double(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length.(states)...)
    times_handle = writeheaders_stime(times_file)

    if length(states[1]) != length(states[2])
        error("Lengths of the two given MPSs do not match!")
    else
        N = length(states[1])
    end

    states = [states...]  # Convert tuple into vector so that we can mutate its elements
    projections = [PH, deepcopy(PH)]  # The projection on different states will be different

    # Prepare for first iteration.
    for (state, PH) in zip(states, projections)
        orthogonalize!(state, 1)
        set_nsite!(PH, 1)
        position!(PH, state, 1)
    end

    current_time = 0.0
    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the state's MPS, not the bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.

                # ha == 1  =>  left-to-right sweep
                # ha == 2  =>  right-to-left sweep
                sweepdir = (ha == 1 ? "right" : "left")
                for (state, PH) in zip(states, projections)
                    tdvp_site_update!(
                        solver,
                        PH,
                        state,
                        site,
                        -0.5Δt; # forward by -im*timestep/2, backwards by im*timestep/2.
                        current_time=(
                            ha == 1 ? current_time + 0.5timestep : current_time + timestep
                        ),
                        sweepdir=sweepdir,
                        which_decomp=decomp,
                        hermitian=hermitian,
                        exp_tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                end
                # At least with TDVP1, `tdvp_site_update!` updates the site at `site`, and
                # leaves the MPS with orthocenter at `site+1` or `site-1` if it sweeping
                # rightwards
                apply!(
                    cb,
                    states...;
                    t=current_time,
                    site=site,
                    sweepend=(ha == 2),
                    sweepdir=sweepdir,
                    alg=TDVP1(),
                )
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", current_time),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maximum(maxlinkdim.(states))),
            ],
        )

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            printoutput_data(io_handle, cb, states...; kwargs...)
            printoutput_ranks(ranks_handle, cb, states...)
            printoutput_stime(times_handle, stime)
        end

        current_time += timestep

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
