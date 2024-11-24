export tdvp1!, adaptivetdvp1!

using ITensors: permute
using ITensorMPS: position!, set_nsite!, check_hascommoninds

"""
    tdvp1!(solver, ψ::MPS, ⃗H::Vector{MPO}, Δt::Number, tf::Number; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i Hⱼ ψₜ`` using the one-site TDVP algorithm,
where `ψ` represents the state of the system and the elements of `⃗H` form… in some way… the
Hamiltonian operator of the system.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `ψ::MPS`: the state of the system.
- `⃗H:Vector{MPO}`: a list of MPOs.
- `Δt::Number`: time step of the evolution.
- `tf::Number`: end time of the evolution.
"""
function tdvp1!(
    solver, psi0::MPS, Hs::Vector{MPO}, time_step::Number, tf::Number; kwargs...
)
    # (Copied from ITensorsTDVP)
    for H in Hs
        check_hascommoninds(siteinds, H, psi0)
        check_hascommoninds(siteinds, H, psi0')
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return tdvp1!(solver, psi0, PHs, time_step, tf; kwargs...)
end

"""
    tdvp1!(solver, ψ::MPS, H::MPO, Δt::Number, tf::Number; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the one-site TDVP algorithm,
where `ψ` represents the state of the system and the elements of `H` is the Hamiltonian
operator of the system.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `ψ::MPS`: the state of the system.
- `H::MPO`: the Hamiltonian operator.
- `tf::Number`: end time of the evolution.
"""
function tdvp1!(solver, state::MPS, H::MPO, timestep::Number, tf::Number; kwargs...)
    return tdvp1!(solver, state, ProjMPO(H), timestep, tf; kwargs...)
end

"""
    tdvp1!(solver, ψ::MPS, ⃗H::Vector{MPO}, Δt::Number, tf::Number; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i Hⱼ ψₜ`` using the one-site TDVP algorithm,
where `ψ` represents the state of the system and the elements of `⃗H` form… in some way… the
Hamiltonian operator of the system.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `ψ::MPS`: the state of the system.
- `PH`: a ProjMPO-like operator encoding the Hamiltonian operator.
- `tf::Number`: end time of the evolution.
"""
function tdvp1!(solver, state::MPS, PH, timestep::Number, tf::Number; kwargs...)
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

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Prepare for first iteration.
    orthogonalize!(state, 1)
    set_nsite!(PH, 1)
    position!(PH, state, 1)

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
                # At least with TDVP1, `tdvp_site_update!` updates the site at `site`, and
                # leaves the MPS with orthocenter at `site+1` or `site-1` if it sweeping
                # rightwards
                apply!(
                    cb,
                    state;
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
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, state)
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

"""
    adaptivetdvp1!(solver, state::MPS, H::Vector{MPO}, Δt::Number, tf::Number; kwargs...)

Like `tdvp1!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.

See [`tdvp1!`](@ref).
"""
function adaptivetdvp1!(
    solver, psi0::MPS, Hs::Vector{MPO}, time_step::Number, tf::Number; kwargs...
)
    # (Copied from ITensorsTDVP)
    for H in Hs
        check_hascommoninds(siteinds, H, psi0)
        check_hascommoninds(siteinds, H, psi0')
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return adaptivetdvp1!(solver, psi0, PHs, time_step, tf; kwargs...)
end

"""
    adaptivetdvp1!(solver, state::MPS, H::MPO, Δt::Number, tf::Number; kwargs...)

Like `tdvp1!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.

See [`tdvp1!`](@ref).
"""
function adaptivetdvp1!(solver, state::MPS, H::MPO, timestep::Number, tf::Number; kwargs...)
    return adaptivetdvp1!(solver, state::MPS, ProjMPO(H), timestep, tf; kwargs...)
end

"""
    adaptivetdvp1!(solver, state::MPS, PH, Δt::Number, tf::Number; kwargs...)

Like `tdvp1!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.

See [`tdvp1!`](@ref).
"""
function adaptivetdvp1!(solver, state::MPS, PH, timestep::Number, tf::Number; kwargs...)
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
    convergence_factor_bonddims = get(kwargs, :convergence_factor_bonddims, 1e-4)
    max_bond = get(kwargs, :max_bond, maxlinkdim(state))
    decomp = get(kwargs, :which_decomp, "qr")

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    Δt = im * timestep
    imag(Δt) == 0 && (Δt = real(Δt))

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Prepare for first iteration
    orthogonalize!(state, 1)
    set_nsite!(PH, 1)
    position!(PH, state, 1)

    current_time = 0.0
    for s in 1:nsteps
        orthogonalize!(state, 1)
        set_nsite!(PH, 1)
        position!(PH, state, 1)

        # Before each sweep, we grow the bond dimensions a bit.
        # See Dunnett and Chin, 2020 [arXiv:2007.13528v2].
        @debug "[Step $s] Attempting to grow the bond dimensions."
        adaptbonddimensions!(state, PH, max_bond, convergence_factor_bonddims)

        stime = @elapsed begin
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                    solver,
                    PH,
                    state,
                    site,
                    -0.5Δt;
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
                apply!(
                    cb,
                    state;
                    t=current_time,
                    bond=site,
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
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, state)
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
