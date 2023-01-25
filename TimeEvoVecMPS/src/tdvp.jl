export tdvp!, tdvpMC!, tdvp1!, tdvp1vec!

using ITensors: position!

zerosite!(PH::ProjMPO) = (PH.nsite = 0)
singlesite!(PH::ProjMPO) = (PH.nsite = 1)
twosite!(PH::ProjMPO) = (PH.nsite = 2)

struct TDVP1 end
struct TDVP2 end

"""
    tdvp!(psi,H::MPO,dt,tf; kwargs...)
Evolve the MPS `psi` up to time `tf` using the two-site time-dependent variational
principle as described in Ref. [1].

# Keyword arguments:
All keyword arguments controlling truncation which are accepted by ITensors.replaceBond!,
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
Unifying time evolution and optimization with matrix product states.
Physical Review B, 94(16).
https://doi.org/10.1103/PhysRevB.94.165116
"""
function tdvp!(psi, H::MPO, dt, tf; kwargs...)
    # This is 2-site TDVP! Shouldn't we name the function accordingly?
    nsteps = Int(tf / dt)
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

    τ = im * dt
    # If dt is imaginary and imag(dt) > 0, this gives us an evolution operator of the
    # form U(dt) = exp(-dt*H) which denotes an "imaginary-time" evolution.
    imag(τ) == 0 && (τ = real(τ))
    # This doesn't just chop off the imaginary part, it also converts the type from
    # Complex{T} to T.

    store_psi0 = get(kwargs, :store_psi0, false)
    store_psi0 && (psi0 = copy(psi))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(psi)
    orthogonalize!(psi, 1)
    PH = ProjMPO(H)
    position!(PH, psi, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (b, ha) in sweepnext(N)
                # Evolve with two-site Hamiltonian.
                twosite!(PH)
                ITensors.position!(PH, psi, b)
                wf = psi[b] * psi[b + 1]
                wf, info = exponentiate(
                    PH, -0.5τ, wf; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim
                )
                
                info.converged == 0 && throw("exponentiate did not converge")
                spec = replacebond!(
                    psi,
                    b,
                    wf;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    kwargs...
                )
                # spec should be the spectrum (aka the singular values?) of the SVD.

                # normalize && ( psi[dir=="left" ? b+1 : b] /= sqrt(sum(eigs(spec))) )

                apply!(
                    cb,
                    psi;
                    t        = s * dt,
                    bond     = b,
                    sweepend = ha == 2,
                    sweepdir = ha == 1 ? "right" : "left",
                    spec     = spec,
                    # This is needed in case cb is a SpecCallback, which needs the
                    # spectrum to compute some quantities, like the entropy.
                    # It is ignored by other types of callback objects.
                    alg      = TDVP2(),
                )

                # Evolve with single-site Hamiltonian backward in time.
                # In the case of imaginary time-evolution this step
                # is not necessary (see Ref. [1])
                i = ha == 1 ? b + 1 : b
                if 1 < i < N && !(dt isa Complex)
                    singlesite!(PH)
                    ITensors.position!(PH, psi, i)
                    psi[i], info = exponentiate(
                        PH,
                        0.5τ,
                        psi[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                elseif i == 1 && dt isa Complex
                    # TODO not sure if this is necessary anymore
                    psi[i] /= sqrt(real(scalar(dag(psi[i]) * psi[i])))
                end
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", dt * s),
                ("dt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(psi)),
            ],
        )

        if dt * s ≈ measurement_ts(cb)[end]
            printoutput_data(io_handle, cb, psi; psi0, kwargs...)
            printoutput_ranks(ranks_handle, cb, psi)
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
    nsteps     = Int(tf / dt)
    cb         = get(kwargs, :callback, NoTEvoCallback())
    hermitian  = get(kwargs, :hermitian, false) # Lindblad superoperator is not Hermitian
    exp_tol    = get(kwargs, :exp_tol, 1e-14)
    krylovdim  = get(kwargs, :krylovdim, 30)
    maxiter    = get(kwargs, :maxiter, 100)
    normalize  = get(kwargs, :normalize, false) # Vectorized states don't need normalization
    io_file    = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc = "Evolving state... ")
    else
        pbar = nothing
    end

    #Preamble
    #println(BLAS.get_config());
    #print("NumThreads: ");

    #println(BLAS.get_num_threads())
    #flush(stdout)

    τ = im * dt
    store_initstate = get(kwargs, :store_initstate, false)
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

    for s = 1:nsteps
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
                ITensors.position!(PH, state, bond)

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
                twositeblock = state[bond] * state[bond+1]
                twositeblock, info = exponentiate(
                    PH,
                    -0.5τ,
                    twositeblock;
                    ishermitian = hermitian,
                    tol = exp_tol,
                    krylovdim = krylovdim,
                )
                info.converged == 0 && throw("exponentiate did not converge")

                # Factorize the twositeblock tensor in two, and replace the ITensors at
                # sites bond and bond + 1 inside the MPS state.
                spec = replacebond!(
                    state,
                    bond,
                    twositeblock;
                    normalize = normalize,
                    ortho = (ha == 1 ? "left" : "right"),
                    kwargs...,
                )
                # normalize && ( state[dir=="left" ? bond+1 : bond] /= sqrt(sum(eigs(spec))) )

                apply!(
                    cb,
                    state;
                    t = s * dt,
                    bond = bond,
                    sweepend = (ha == 2),
                    # apply! does nothing if sweepend is false, so this way we are doing
                    # the measurement only on the second sweep, from right to left.
                    sweepdir = (ha == 1 ? "right" : "left"),
                    spec = spec,
                    alg = TDVP2(),
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
                    singlesite!(PH)
                    ITensors.position!(PH, state, i)
                    state[i], info = exponentiate(
                        PH,
                        0.5τ,
                        state[i];
                        ishermitian = hermitian,
                        tol = exp_tol,
                        krylovdim = krylovdim,
                        maxiter = maxiter,
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
            showvalues = [
                ("t", dt * s),
                ("dt step time", round(stime; digits = 3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if dt * s ≈ measurement_ts(cb)[end]
            printoutput_data(io_handle, cb, state; initstate, kwargs...)
            printoutput_ranks(ranks_handle, cb, state)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file)    && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end

function tdvp1!(state, H::MPO, Δt, tf; kwargs...)
    nsteps = Int(tf / Δt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_state0, false)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    imag(Δt) == 0 && (Δt = real(Δt))

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Prepare for first iteration
    orthogonalize!(state, 1)
    PH = ProjMPO(H)
    singlesite!(PH)
    position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the state's MPS, not the bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                # 
                # Create a one-site projection of the Hamiltonian on the first site.
                singlesite!(PH)
                ITensors.position!(PH, state, site)

                # TODO Here we could merge TDVP1 and TDVP2.
                # φ = state[site]

                # DEBUG
                #println("Forward time evolution: site, ha ",site, ha);

                #exptime = @elapsed begin
                φ, info = exponentiate(
                    PH,
                    -0.5Δt,
                    state[site];
                    ishermitian=hermitian,
                    tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                    eager=true,
                )
                #end
                #println("Forward exponentiation elapsed time: ", exptime)
                info.converged == 0 && throw("exponentiate did not converge")

                # Replace (temporarily) the local tensor with the evolved one.
                state[site] = φ

                # Calculate the expectation values of the observables within cb.
                # TODO Check if measurement is faulty.
                #mtime = @elapsed begin
                apply!(
                    cb,
                    state;
                    t        = s * Δt,
                    bond     = site,
                    sweepend = ha == 2,
                    sweepdir = ha == 1 ? "right" : "left",
                    #spec=spec,
                    alg=TDVP1(),
                )
                #end mtime
                #println("Tempo per misure: ", mtime)

                # Copypasted from
                #https://github.com/ITensor/ITensorTDVP.jl/blob/main/src/tdvp_step.jl
                if (ha == 1 && (site != N)) || (ha == 2 && site != 1)
                    # Start from a right-canonical MPS A, where each matrix A(n) = Aᵣ(n) is
                    # right-orthogonal.
                    # Start at site n = 1 and repeat the following steps:
                    # 1. Evolve A(n) for a time step Δt.
                    # 2. Factorize the updated A(n) as Aₗ(n)C(n) such that the matrix Aₗ,
                    # which will be left at site n, is left-orthogonal.
                    # 3. Evolve C(n) backwards in time according for a time step Δt, before
                    # absorbing it into the next site to create A(n + 1) = C(n)Aᵣ(n + 1).
                    # 4. Evolve A(n + 1) and so on...
                    #
                    b1 = (ha == 1 ? site + 1 : site) # ???
                    Δb = (ha == 1 ? +1 : -1)
                    # site + Δb is the physical index of the next site in the sweep.
                    uinds = uniqueinds(φ, state[site + Δb])

                    #svdtime = @elapsed begin
                    U, S, V = svd(φ, uinds)
                    #end
                    #println("Tempo svd: ", svdtime)

                    state[site] = U # This is left(right)-orthogonal if ha==1(2).
                    phi0 = S * V
                    if ha == 1
                        ITensors.setleftlim!(state, site)
                    elseif ha == 2
                        ITensors.setrightlim!(state, site)
                    end

                    zerosite!(PH)
                    #postime = @elapsed begin
                    position!(PH, state, b1)
                    #end
                    #println("tempo position site+1", postime)

                    #Debug
                    #println("Backward-time site, ha", site, ha)
                    #exptime = @elapsed begin
                    phi0, info = exponentiate(
                        PH,
                        0.5Δt,
                        phi0;
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                        eager=true,
                    )
                    #end
                    #println("Tempo esponenziazione indietro: ", exptime)
                    #println("backward done!")

                    # Reunite the backwards-evolved C(site) with the matrix on the
                    # next site.
                    state[site + Δb] = phi0 * state[site + Δb]

                    if ha == 1
                        ITensors.setrightlim!(state, site + Δb + 1)
                    elseif ha == 2
                        ITensors.setleftlim!(state, site + Δb - 1)
                    end

                    singlesite!(PH)
                end

                # Adesso il passo di evoluzione e` finito.
                # Considerando che misuriamo "durante" lo sweep
                # a sx, adesso l'ortogonality center e` sul sito a sx del bond
                # quindi credo che la misura vada spostata piu` su
                # ovvero prima di rendere il tensore right orthogonal
            end
        end

        #println("time-step time: ",s*dt," t=", stime)
        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", Δt * s),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if Δt * s ≈ measurement_ts(cb)[end]
            printoutput_data(io_handle, cb, state; state0, kwargs...)
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

"""
    tdvp1vec!(state, H::MPO, Δt, tf; kwargs...)

For vectorized state it is still unclear whether the measurements can be made before
the sweep is complete. Therefore, until this question gets an answer, this function
postpones the measurements of all observables until all the sites of the state are
updated.
"""
function tdvp1vec!(state, H::MPO, Δt, tf; kwargs...)
    nsteps = Int(tf / Δt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_state0, false)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    imag(Δt) == 0 && (Δt = real(Δt))

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Prepare for first iteration
    orthogonalize!(state, 1)
    PH = ProjMPO(H)
    singlesite!(PH)
    position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the state's MPS, not the bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                # 
                # Create a one-site projection of the Hamiltonian on the first site.
                singlesite!(PH)
                ITensors.position!(PH, state, site)

                # TODO Here we could merge TDVP1 and TDVP2.
                # φ = state[site]

                # DEBUG
                #println("Forward time evolution: site, ha ",site, ha);

                #exptime = @elapsed begin
                φ, info = exponentiate(
                    PH,
                    -0.5Δt,
                    state[site];
                    ishermitian=hermitian,
                    tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                    eager=true,
                )
                #end
                #println("Forward exponentiation elapsed time: ", exptime)
                info.converged == 0 && throw("exponentiate did not converge")

                # Replace (temporarily) the local tensor with the evolved one.
                state[site] = φ

                # Copypasted from
                #https://github.com/ITensor/ITensorTDVP.jl/blob/main/src/tdvp_step.jl
                if (ha == 1 && (site != N)) || (ha == 2 && site != 1)
                    # Start from a right-canonical MPS A, where each matrix A(n) = Aᵣ(n) is
                    # right-orthogonal.
                    # Start at site n = 1 and repeat the following steps:
                    # 1. Evolve A(n) for a time step Δt.
                    # 2. Factorize the updated A(n) as Aₗ(n)C(n) such that the matrix Aₗ,
                    # which will be left at site n, is left-orthogonal.
                    # 3. Evolve C(n) backwards in time according for a time step Δt, before
                    # absorbing it into the next site to create A(n + 1) = C(n)Aᵣ(n + 1).
                    # 4. Evolve A(n + 1) and so on...
                    #
                    b1 = (ha == 1 ? site + 1 : site) # ???
                    Δb = (ha == 1 ? +1 : -1)
                    # site + Δb is the physical index of the next site in the sweep.
                    uinds = uniqueinds(φ, state[site + Δb])

                    #svdtime = @elapsed begin
                    U, S, V = svd(φ, uinds)
                    #end
                    #println("Tempo svd: ", svdtime)

                    state[site] = U # This is left(right)-orthogonal if ha==1(2).
                    phi0 = S * V
                    if ha == 1
                        ITensors.setleftlim!(state, site)
                    elseif ha == 2
                        ITensors.setrightlim!(state, site)
                    end

                    zerosite!(PH)
                    #postime = @elapsed begin
                    position!(PH, state, b1)
                    #end
                    #println("tempo position site+1", postime)

                    #Debug
                    #println("Backward-time site, ha", site, ha)
                    #exptime = @elapsed begin
                    phi0, info = exponentiate(
                        PH,
                        0.5Δt,
                        phi0;
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                        eager=true,
                    )
                    #end
                    #println("Tempo esponenziazione indietro: ", exptime)
                    #println("backward done!")

                    # Reunite the backwards-evolved C(site) with the matrix on the
                    # next site.
                    state[site + Δb] = phi0 * state[site + Δb]

                    if ha == 1
                        ITensors.setrightlim!(state, site + Δb + 1)
                    elseif ha == 2
                        ITensors.setleftlim!(state, site + Δb - 1)
                    end

                    singlesite!(PH)
                end

                # Adesso il passo di evoluzione e` finito.
                # Considerando che misuriamo "durante" lo sweep
                # a sx, adesso l'ortogonality center e` sul sito a sx del bond
                # quindi credo che la misura vada spostata piu` su
                # ovvero prima di rendere il tensore right orthogonal
            end
        end

        # Now the backwards sweep has ended, so the whole MPS of the state is up-to-date.
        # We can then calculate the expectation values of the observables within cb.
        for site in 1:N
            apply!(
                   cb,
                   state;
                   t        = s * Δt,
                   bond     = site,
                   sweepend = true,
                   sweepdir = "right", # The value doesn't matter
                   alg=TDVP1(),
                  )
        end

        #println("time-step time: ",s*dt," t=", stime)
        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", Δt * s),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if Δt * s ≈ measurement_ts(cb)[end]
            printoutput_data(io_handle, cb, state; state0, kwargs...)
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

"""
    writeheaders_data(io_file, cb; kwargs...)

Prepare the output file `io_file`, writing the column headers for storing the data of
the observables defined in `cb`, the time steps, and other basic quantities.
"""
function writeheaders_data(io_file, cb; kwargs...)
    io_handle = nothing
    if !isnothing(io_file)
        io_handle = open(io_file, "w")
        @printf(io_handle, "#%19s", "time")
        res = measurements(cb)
        for o in sort(collect(keys(res)))
            @printf(io_handle, "%40s", o)
        end
        if get(kwargs, :store_state0, false)
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
        @printf(ranks_handle, "#%19s", "time")
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
        @printf(times_handle, "#%19s", "walltime (sec)")
        @printf(times_handle, "\n")
    end

    return times_handle
end

function printoutput_data(io_handle, cb, psi; psi0, kwargs...)
    if !isnothing(io_handle) && !isempty(measurement_ts(cb))
        results = measurements(cb)
        @printf(io_handle, "%40.15f", measurement_ts(cb)[end])
        for o in sort(collect(keys(results)))
            @printf(io_handle, "%40.15f", results[o][end][1])
        end

        if store_psi0
            overlap = dot(psi0, psi)
            @printf(io_handle, "%40s.15f%40.15f", real(overlap), imag(overlap))
        end

        # Print the norm
        @printf(io_handle, "40.15f", norm(psi))
        @printf(io_handle, "\n")
        flush(io_handle)
    end

    return nothing
end

function printoutput_ranks(ranks_handle, cb, state)
    if !isnothing(ranks_handle) && !isempty(measurement_ts(cb))
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
    if !isnothing(times_handle) && !isempty(measurement_ts(cb))
        @printf(times_handle, "%20.4f\n", stime)
        flush(times_handle)
    end

    return nothing
end
