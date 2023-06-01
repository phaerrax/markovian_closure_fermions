export tdvp2!, tdvpMC!

using ITensors: position!, set_nsite!

"""
    tdvp2!(ψ, H::MPO, timestep, endtime; kwargs...)
Evolve the MPS `ψ` up to time `endtime` using the two-site time-dependent variational
principle as described in [1].

# Keyword arguments:
All keyword arguments controlling truncation which are accepted by ITensors.replacebond!,
namely:
- `maxdim::Int`: If specified, keep only `maxdim` largest singular values after
applying the gate.
- `mindim::Int`: Minimal number of singular values to keep if truncation is performed
according to value specified by `cutoff`.
- `cutoff::Float`: If specified, keep the minimal number of singular values such that the
discarded weight is smaller than `cutoff` (but the bond dimension will be kept smaller
than `maxdim`).
- `absoluteCutoff::Bool`: If `true` truncate all singular-values whose square is smaller
than `cutoff`.

In addition the following keyword arguments are supported:
- `hermitian::Bool` (`true`) : whether the MPO `H` represents an Hermitian operator.
This will be passed to the Krylov exponentiation routine (`KrylovKit.exponentiate`) which
will in turn use a Lancosz algorithm in the case of an hermitian operator.
- `exp_tol::Float` (1e-14) : The error tolerance for `KrylovKit.exponentiate` (note that
default value was not optimized yet, so you might want to play around with it).
- `progress::Bool` (`true`) : If `true` a progress bar will be displayed

# References:
[1] Haegeman, J., Lubich, C., Oseledets, I., Vandereycken, B., & Verstraete, F. (2016).
“Unifying time evolution and optimization with matrix product states”
Physical Review B, 94(16).
https://doi.org/10.1103/PhysRevB.94.165116
"""
function tdvp2!(ψ, H::MPO, timestep, endtime; kwargs...)
    nsteps = Int(endtime / timestep)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)

    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    Δt = im * timestep
    # If `timestep` is imaginary and imag(timestep) > 0, this gives us an evolution operator
    # of the form U(Δt) = exp(-Δt H) which denotes an "imaginary-time" evolution.
    imag(Δt) == 0 && (Δt = real(Δt))
    # A unitary evolution is associated to a real `timestep`. In this case, Δt is purely
    # imaginary, as it should.
    # Otherwise, with an imaginary-time evolution, Δt is real, but the Type of the variable
    # is Complex, so we truncate any imaginary part away.
    # (`real` doesn't just chop off the imaginary part, it also converts the type from
    # Complex{T} to T.)

    store_psi0 = get(kwargs, :store_psi0, false)
    store_psi0 && (psi0 = copy(ψ))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(ψ)
    orthogonalize!(ψ, 1)
    # Move the orthogonality centre to site 1, i.e. make ψ right-canonical.
    PH = ProjMPO(H)
    position!(PH, ψ, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (b, ha) in sweepnext(N)
                # 1. Evolve with two-site Hamiltonian.
                #    ---------------------------------
                #    We project the Hamiltonian on the current bond b and the next one,
                #    then we evolve the (b, b+1) block for half a time step.
                #    The sweepnext iterator takes care of the correct indices: the pair
                #    (b, ha) here takes the values
                #       (1, 1), …, (N-1, 1), (N-1, 2), …, (1, 2)
                #    so that the correct pair of bonds is always (b, b+1).
                twosite!(PH)
                position!(PH, ψ, b)
                wf = ψ[b] * ψ[b + 1]
                wf, info = exponentiate(
                    PH, -0.5Δt, wf; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim
                )

                info.converged == 0 && throw("exponentiate did not converge")
                # Replace the ITensors of the MPS `ψ` at sites b and b+1 with `wf`,
                # which is factorized according to the orthogonalization specification
                # given by `ortho` (left for a left-to-right sweep, right otherwise).
                # (replacebond! normalizes the result, since we pass :normalize="true"
                # within the kwargs).
                spec = replacebond!(
                    ψ,
                    b,
                    wf;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    kwargs...,
                )
                # spec is the spectrum (aka the singular values?) of the SVD.
                # Some types of callback objects might need it later, in order to compute
                # the entropy or other related quantities.

                # 2. Measure the observables.
                #    ------------------------
                #    When we are sweeping right-to-left, once the block at sites (b, b+1)
                #    has been evolved, the tensor at ψ[b + 1] has completed its evolution
                #    within the time step dt.
                #    The MPS is
                #    • left-orthogonal from ψ[1] to ψ[b - 1]
                #    • right-orthogonal from ψ[b] to ψ[end]
                #    so this is a good time to measure observables that are local to
                #    site b+1: when contracting in inner(ψ', A(n), ψ), all the sites
                #    left of ψ[b] (excluded) give the identity, and so do all those
                #    right of ψ[b + 1].
                #    The measurement can then be performed using the tensor composed by
                #    only ψ[b] and ψ[b + 1].
                apply!(
                    cb,
                    ψ;
                    t=s * timestep,
                    # This is only for storage purposes; we need the original `timestep`.
                    bond=b,
                    sweepend=(ha == 2), # apply! is skipped if ha == 1
                    sweepdir=(ha == 1 ? "right" : "left"),
                    spec=spec,
                    alg=TDVP2(),
                )

                # 3. Evolve with single-site Hamiltonian backward in time.
                #    -----------------------------------------------------
                #    Evolve the "next" block backwards for half a time step.
                #    Which block is the "next" block depends on the direction of the sweep:
                #    • when sweeping left-to-right, the pivot is on site `b`, and the next
                #      tensor is the one to the right, so `b+1`;
                #    • when sweeping right-to-left, the pivot is on site `b+1`, and the
                #      next tensor is the one to the left, so `b`.
                #    This step is not necessary in the case of imaginary time-evolution [1].
                i = ha == 1 ? b + 1 : b
                if 1 < i < N && !(dt isa Complex)
                    set_nsite!(PH, 1)
                    position!(PH, ψ, i)
                    ψ[i], info = exponentiate(
                        PH,
                        0.5Δt,
                        ψ[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                elseif i == 1 && dt isa Complex
                    # dt isa Complex <==> imaginary-time evolution.
                    # TODO not sure if this is necessary anymore
                    ψ[i] /= sqrt(real(scalar(dag(ψ[i]) * ψ[i])))
                end
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", timestep * s),
                ("dt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(ψ)),
            ],
        )

        if !isempty(measurement_ts(cb)) && timestep * s ≈ measurement_ts(cb)[end]
            if store_psi0
                printoutput_data(io_handle, cb, ψ; psi0=psi0, kwargs...)
            else
                printoutput_data(io_handle, cb, ψ; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, ψ)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end

"""
    tdvpMC!(state, H::MPO, dt, tf; kwargs...)

Evolve the MPS `state` using the MPO `H` from 0 to `tf` using an integration step `dt`.
"""
function tdvpMC!(state, H::MPO, dt, tf; kwargs...)
    nsteps = Int(tf / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, false) # Lindblad superoperator is not Hermitian
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false) # Vectorized states don't need normalization
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    #Preamble
    #println(BLAS.get_config());
    #print("NumThreads: ");

    #println(BLAS.get_num_threads())
    #flush(stdout)

    τ = im * dt
    store_initstate = get(kwargs, :store_psi0, false)
    imag(τ) == 0 && (τ = real(τ))

    # Copy the initial state if store_initstate is true
    store_initstate && (initstate = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)
    orthogonalize!(state, 1)
    PH = ProjMPO(H)
    position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (bond, ha) in sweepnext(N)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                #
                # 1st step
                # --------
                # Evolve using two-site Hamiltonian: we extract the tensor on sites (j,j+1)
                # and we solve locally the equations of motion, then we put the result in
                # place of the original tensor.
                #
                # We set PH.nsite to 2, since we want to use a two-site update method,
                # and we shift the projection PH of H such that the set of unprojected
                # sites begins at site bond.
                twosite!(PH)
                position!(PH, state, bond)

                # Completing a single left-to-right sweep is a first-order integrator that
                # produces an updated |ψ⟩ at time t + τ with a local integration error of
                # order O(τ²). Completing the right-to-left sweep is equivalent to
                # composing this integrator with its adjoint, resulting in a second-order
                # symmetric method so that the state at time t + 2τ has a more favourable
                # error of order O(τ³). It is thus natural to set τ → τ/2 and to define
                # the complete sweep (left and right) as a single integration step.
                # [`Unifying time evolution and optimization with matrix product states'.
                # Jutho Haegeman, Christian Lubich, Ivan Oseledets, Bart Vandereycken and
                # Frank Verstraete, Physical Review B 94, 165116, p. 10 October 2016]
                twositeblock = state[bond] * state[bond + 1]
                twositeblock, info = exponentiate(
                    PH,
                    -0.5τ,
                    twositeblock;
                    ishermitian=hermitian,
                    tol=exp_tol,
                    krylovdim=krylovdim,
                )
                info.converged == 0 && throw("exponentiate did not converge")

                # Factorize the twositeblock tensor in two, and replace the ITensors at
                # sites bond and bond + 1 inside the MPS state.
                spec = replacebond!(
                    state,
                    bond,
                    twositeblock;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    kwargs...,
                )
                # normalize && ( state[dir=="left" ? bond+1 : bond] /= sqrt(sum(eigs(spec))) )

                apply!(
                    cb,
                    state;
                    t=s * dt,
                    bond=bond,
                    sweepend=(ha == 2),
                    # apply! does nothing if sweepend is false, so this way we are doing
                    # the measurement only on the second sweep, from right to left.
                    sweepdir=(ha == 1 ? "right" : "left"),
                    spec=spec,
                    alg=TDVP2(),
                )

                # 2nd step
                # --------
                # Evolve the second site of the previous two using the single-site
                # Hamiltonian, backward in time.
                # In the case of imaginary time-evolution this step is not necessary (see
                # Ref. 1).
                i = (ha == 1 ? bond + 1 : bond)
                # The "second site" is the one in the direction of the sweep.
                if 1 < i < N && !(dt isa Complex)
                    set_nsite!(PH, 1)
                    position!(PH, state, i)
                    state[i], info = exponentiate(
                        PH,
                        0.5τ,
                        state[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                elseif i == 1 && dt isa Complex
                    # Normalization is not necessary (or even wrong...) if the state
                    # is a density matrix.
                    #state[i] /= sqrt(real(scalar(dag(state[i]) * state[i])))
                end
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", dt * s),
                ("dt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && dt * s ≈ measurement_ts(cb)[end]
            if store_initstate
                printoutput_data(io_handle, cb, state; psi0=initstate, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, state)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
