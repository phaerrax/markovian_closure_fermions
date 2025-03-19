using LinearAlgebra,
    TEDOPA, MarkovianClosure, OpenSystemsChainMapping, DifferentialEquations, OffsetArrays

function tedopa_chain_coefficients(μ, T, NE; kwargs...)
    d = Dict{AbstractString,Any}(
        "chain_length" => round(Int, 1.5 * NE),
        "PolyChaos_nquad" => get(kwargs, :nquad, 2 * NE),
    )
    env = Dict(
        "spectral_density_function" => "1/10pi * sqrt((x+a[1]) * (2-(x+a[1])))",
        "domain" => [-μ, 2 - μ],
        "chemical_potential" => 0,
        "spectral_density_parameters" => [μ],
        "temperature" => T,
    )
    push!(d, "environment" => env)
    cfs = chainmapping_thermofield(d)

    return cfs
end

function mc_ode(tf, ε, sysinit, μ, T, NE, NC; generated_chain_length=200, kwargs...)
    cfs = tedopa_chain_coefficients(μ, T, generated_chain_length; kwargs...)

    envL = ModeChain(
        1:length(cfs[:filled].frequencies),
        cfs[:filled].frequencies,
        cfs[:filled].couplings[2:end],
    )
    envR = ModeChain(
        1:length(cfs[:empty].frequencies),
        cfs[:empty].frequencies,
        cfs[:empty].couplings[2:end],
    )

    trunc_envL, mcL = markovianclosure(
        envL, NC, NE; asymptoticfrequency=1 - μ, asymptoticcoupling=2 / 4
    )
    trunc_envR, mcR = markovianclosure(
        envR, NC, NE; asymptoticfrequency=1 - μ, asymptoticcoupling=2 / 4
    )

    ω = (L=trunc_envL.frequencies, R=trunc_envR.frequencies)
    η = (L=cfs[:filled].couplings[1], R=cfs[:empty].couplings[1])
    κ = (L=trunc_envL.couplings, R=trunc_envR.couplings)

    ν = (L=freqs(mcL), R=freqs(mcR))
    λ = (L=-innercoups(mcL), R=innercoups(mcR))
    ζ = (L=outercoups(mcL), R=outercoups(mcR))
    γ = (L=damps(mcL), R=damps(mcR))

    f = [reverse(ν.L); reverse(ω.L); ε; ω.R; ν.R]

    g = [
        conj(reverse(λ.L))
        0
        conj(reverse(κ.L))
        conj(η.L)
        η.R
        κ.R
        0
        λ.R
    ]
    # Tridiagonal([1,1], [2,2,2], [3,3]) == diagm(-1 => [1,1], 0 => [2,2,2], 1 => [3,3])
    H = Tridiagonal(conj(g), f, g)

    K = OffsetArray(
        zeros(ComplexF64, size(H)), ((-(NC + NE)):(NC + NE), (-(NC + NE)):(NC + NE))
    )
    for k in 1:NC
        K[-NE - k, -NE] = ζ.L[k]
        K[-NE, -NE - k] = conj(ζ.L[k])
        K[NE + k, NE] = ζ.R[k]
        K[NE, NE + k] = conj(ζ.R[k])
    end
    H += OffsetArrays.no_offset_view(K)
    # remove offsets, otherwise matrix multiplication doesn't work.

    M = 2(NC + NE) + 1
    Γ = (L=Diagonal([reverse(γ.L); zeros(M - NC)]), R=Diagonal([zeros(M - NC); γ.R]))
    S = (L=Diagonal([ones(NC); zeros(M - NC)]), R=Diagonal([zeros(M - NC); ones(NC)]))

    X = im * H
    Γs = Γ.L .+ Γ.R

    eqmode = get(kwargs, :eqmode, :linear)
    if eqmode == :linear
        function lindblad_linear!(dA, A, p, t)
            dA[1:M, :] .=
            # XA + AX'
                X * A[1:M, :] .+ A[1:M, :] * X' .+
                # 1/2 (ΓL (A+B) SL + SL (A+B) ΓL)
                0.5 .* Γ[:L] * (A[1:M, :] .+ A[(M + 1):end, :]) * S[:L] .+
                0.5 .* S[:L] * (A[1:M, :] .+ A[(M + 1):end, :]) * Γ[:L] .+
                # ΓR A SR + SR A ΓR
                S[:R] * A[1:M, :] * Γ[:R] .+ Γ[:R] * A[1:M, :] * S[:R] .+
                # -1/2 ((ΓL+ΓR) A + A (ΓL+ΓR))
                -0.5 .* (Γs * A[1:M, :] .+ A[1:M, :] * Γs)

            dA[(M + 1):end, :] .= -transpose(dA[1:M, :])
            return nothing
        end

        A0 = zeros(ComplexF64, 2M, M)
        A0[1:M, :] .= diagm(
            0 => [ones(NC + NE); (sysinit == "Occ" ? 1 : 0); zeros(NC + NE)]
        )
        A0[(M + 1):end, :] .= I - transpose(A0[1:M, :])

        ode = ODEProblem(lindblad_linear!, A0, (0.0, tf))
        return ode
    elseif eqmode == :affine
        function lindblad_affine!(dA, A, p, t)
            dA .=
            # XA + AX'
                X * A .+ A * X' .+
                # 1/2 (ΓL (A-A^t) SL + SL (A-A^t) ΓL)
                0.5 .* (
                    Γ[:L] * (A .- transpose(A)) * S[:L] .+
                    S[:L] * (A .- transpose(A)) * Γ[:L]
                ) .+
                # 1/2 (ΓL SL + SL ΓL)
                0.5 .* (Γ[:L] * S[:L] .+ S[:L] * Γ[:L]) .+
                # ΓR A SR + SR A ΓR
                S[:R] * A * Γ[:R] .+ Γ[:R] * A * S[:R] .+
                # -1/2 ((ΓL+ΓR) A + A (ΓL+ΓR))
                -0.5 .* (Γs * A .+ A * Γs)
            return nothing
        end

        A0 = zeros(ComplexF64, M, M)
        A0 .= diagm(0 => [ones(NC + NE); (sysinit == "Occ" ? 1 : 0); zeros(NC + NE)])
        ode = ODEProblem(lindblad_affine!, A0, (0.0, tf))
        return ode
    elseif eqmode == :splitcomplex
        Xr, Xi = reim(-im .* X)
        function lindblad_splitcomplex!(dA, A, p, t)
            # We double the matrix and store real and imaginary separately. Some of the
            # equations need to be changed, but only the commutator part, since the rest
            # is made up of real matrices.
            # real part: A[:, 1:M]
            # imag part: A[:, (M+1):end]
            dA[:, 1:M] .=
            # XA + AX'
                -Xr * A[:, (M + 1):end] .- Xi * A[:, 1:M] .- A[:, 1:M] * Xi .+
                A[:, (M + 1):end] * Xr .+
                # 1/2 (ΓL (A-A^t) SL + SL (A-A^t) ΓL)
                0.5 .* (
                    Γ[:L] * (A[:, 1:M] .- transpose(A[:, 1:M])) * S[:L] .+
                    S[:L] * (A[:, 1:M] .- transpose(A[:, 1:M])) * Γ[:L]
                ) .+
                # 1/2 (ΓL SL + SL ΓL)
                0.5 .* (Γ[:L] * S[:L] .+ S[:L] * Γ[:L]) .+
                # ΓR A SR + SR A ΓR
                S[:R] * A[:, 1:M] * Γ[:R] .+ Γ[:R] * A[:, 1:M] * S[:R] .+
                # -1/2 ((ΓL+ΓR) A + A (ΓL+ΓR))
                -0.5 .* (Γs * A[:, 1:M] .+ A[:, 1:M] * Γs)

            dA[:, (M + 1):end] .=
            # XA + AX'
                Xr * A[:, 1:M] .- Xi * A[:, (M + 1):end] .- A[:, 1:M] * Xr .-
                A[:, (M + 1):end] * Xi .+
                # 1/2 (ΓL (A-A^t) SL + SL (A-A^t) ΓL)
                0.5 .* (
                    Γ[:L] * (A[:, (M + 1):end] .- transpose(A[:, (M + 1):end])) * S[:L] .+
                    S[:L] * (A[:, (M + 1):end] .- transpose(A[:, (M + 1):end])) * Γ[:L]
                ) .+
                # 1/2 (ΓL SL + SL ΓL)
                # (imaginary part of this term is zero)
                # ΓR A SR + SR A ΓR
                S[:R] * A[:, (M + 1):end] * Γ[:R] .+ Γ[:R] * A[:, (M + 1):end] * S[:R] .+
                # -1/2 ((ΓL+ΓR) A + A (ΓL+ΓR))
                -0.5 .* (Γs * A[:, (M + 1):end] .+ A[:, (M + 1):end] * Γs)
            return nothing
        end

        A0 = zeros(M, 2M)
        A0[:, 1:M] .= diagm(
            0 => [ones(NC + NE); (sysinit == "Occ" ? 1 : 0); zeros(NC + NE)]
        )
        ode = ODEProblem(lindblad_splitcomplex!, A0, (0.0, tf))
        return ode
    end
end

function tedopa_ode(tf, ε, sysinit, μ, T, NE; kwargs...)
    cfs = tedopa_chain_coefficients(μ, T, NE; kwargs...)

    envL = ModeChain(
        1:length(cfs[:filled].frequencies),
        cfs[:filled].frequencies,
        cfs[:filled].couplings[2:end],
    )
    envR = ModeChain(
        1:length(cfs[:empty].frequencies),
        cfs[:empty].frequencies,
        cfs[:empty].couplings[2:end],
    )
    trunc_envL = first(envL, NE)
    trunc_envR = first(envR, NE)

    ω = (L=trunc_envL.frequencies, R=trunc_envR.frequencies)
    η = (L=cfs[:filled].couplings[1], R=cfs[:empty].couplings[1])
    κ = (L=trunc_envL.couplings, R=trunc_envR.couplings)

    f = [reverse(ω.L); ε; ω.R]
    g = [
        conj(reverse(κ.L))
        conj(η.L)
        η.R
        κ.R
    ]
    H = Tridiagonal(conj(g), f, g)

    function lindblad!(dA, A, p, t)
        dA .= im .* (H * A .- A * H')
        return nothing
    end

    A0 = diagm(
        0 => [ones(ComplexF64, NE); (sysinit == "Occ" ? 1 : 0); zeros(ComplexF64, NE)]
    )

    ode = ODEProblem(lindblad!, A0, (0.0, tf))
    return ode
end

function simulate(
    mc_solver; tf, NE=10, NC=6, μ=0.2, T=0.4, ε=-π / 8, eqmode=:linear, kwargs...
)
    NE_tedopa = round(Int, 1.2 * tf)
    mc = mc_ode(tf, ε, "Occ", μ, T, NE, NC; eqmode, kwargs...)
    tedopa = tedopa_ode(tf, ε, "Occ", μ, T, NE_tedopa)

    sol_mc = solve(mc, mc_solver; kwargs...)
    sol_tedopa = solve(tedopa; kwargs...)

    return (mc=sol_mc, tedopa=sol_tedopa)
end

function simulate(; tf, NE=10, NC=6, μ=0.2, T=0.4, ε=-π / 8, eqmode=:linear, kwargs...)
    NE_tedopa = round(Int, 1.2 * tf)
    mc = mc_ode(tf, ε, "Occ", μ, T, NE, NC; eqmode)
    tedopa = tedopa_ode(tf, ε, "Occ", μ, T, NE_tedopa)

    sol_mc = solve(mc; kwargs...)
    sol_tedopa = solve(tedopa; kwargs...)

    return (mc=sol_mc, tedopa=sol_tedopa)
end

function plot_solution(sol_mc, sol_tedopa, n, m=n; NC=6)
    NE_tedopa = div(size(sol_tedopa[:, :, 1], 2), 2)
    NE_mc = div(minimum(size(sol_mc[:, :, 1])), 2) - NC

    n_tedopa = real.(sol_tedopa[NE_tedopa + 1 + n, NE_tedopa + 1 + m, :])
    n_mc = real.(sol_mc[NE_mc + NC + 1 + n, NE_mc + NC + 1 + m, :])

    ymin = max(minimum(n_tedopa), minimum(n_mc), -1)
    ymax = min(maximum(n_tedopa), maximum(n_mc), 1)

    p = plot(; title="⟨n_$n(t)⟩", xlabel="t", ylim=(ymin, ymax))
    plot!(p, sol_mc.t, n_mc; label="MC")
    plot!(p, sol_tedopa.t, n_tedopa; label="TEDOPA")
    return p
end

function mcsf_setting(ε, sysinit, μ, T, NE, NC; generated_chain_length=200, kwargs...)
    cfs = tedopa_chain_coefficients(μ, T, NE; generated_chain_length, kwargs...)

    envL = ModeChain(
        1:length(cfs[:filled].frequencies),
        cfs[:filled].frequencies,
        cfs[:filled].couplings[2:end],
    )
    envR = ModeChain(
        1:length(cfs[:empty].frequencies),
        cfs[:empty].frequencies,
        cfs[:empty].couplings[2:end],
    )

    trunc_envL, mcL = markovianclosure(
        envL, NC, NE; asymptoticfrequency=1 - μ, asymptoticcoupling=2 / 4
    )
    trunc_envR, mcR = markovianclosure(
        envR, NC, NE; asymptoticfrequency=1 - μ, asymptoticcoupling=2 / 4
    )

    ω = (L=trunc_envL.frequencies, R=trunc_envR.frequencies)
    η = (L=cfs[:filled].couplings[1], R=cfs[:empty].couplings[1])
    κ = (L=trunc_envL.couplings, R=trunc_envR.couplings)

    ν = (L=freqs(mcL), R=freqs(mcR))
    λ = (L=-innercoups(mcL), R=innercoups(mcR))
    ζ = (L=outercoups(mcL), R=outercoups(mcR))
    γ = (L=damps(mcL), R=damps(mcR))

    f = [reverse(ν.L); reverse(ω.L); ε; ω.R; ν.R]
    g = [
        conj(reverse(λ.L))
        0
        conj(reverse(κ.L))
        conj(η.L)
        η.R
        κ.R
        0
        λ.R
    ]
    H = Tridiagonal(conj(g), f, g)

    K = OffsetArray(
        zeros(ComplexF64, size(H)), ((-(NC + NE)):(NC + NE), (-(NC + NE)):(NC + NE))
    )
    for k in 1:NC
        K[-NE, -NE - k] = ζ.L[k]
        K[-NE - k, -NE] = conj(ζ.L[k])
        K[NE + k, NE] = conj(ζ.R[k])
        K[NE, NE + k] = ζ.R[k]
    end
    H += OffsetArrays.no_offset_view(K)
    # remove offsets, otherwise matrix multiplication doesn't work.

    M = 2(NC + NE) + 1
    Γ = (L=Diagonal([reverse(γ.L); zeros(M - NC)]), R=Diagonal([zeros(M - NC); γ.R]))

    X = zeros(Complex{Float64}, 2M, 2M)
    X[1:M, 1:M] .= -im .* H .+ 0.5 .* (Γ.L .- Γ.R)
    X[(M + 1):end, (M + 1):end] .= -im .* H' .- 0.5 .* (Γ.L .- Γ.R)
    X[1:M, (M + 1):end] .= Γ.L
    X[(M + 1):end, 1:M] .= Γ.R

    c₀ = zeros(size(X))
    c₀[1:M, 1:M] .= Diagonal([ones(NE + NC); (sysinit == "Occ" ? 1 : 0); zeros(NE + NC)])
    c₀[(M + 1):end, (M + 1):end] .= I - c₀[1:M, 1:M]
    c₀[1:M, (M + 1):end] .= c₀[1:M, 1:M]
    c₀[(M + 1):end, 1:M] .= I - c₀[1:M, 1:M]

    return X, c₀
    #=
    Apparently this X is always diagonalisable. Remember that

    evals, evecs = eigen(X)
    evecs * Diagonal(evals) * inv(evecs) ≈ X
    =#
end

function evolve_sf_correlation_matrix(ts, generator, initialmatrix)
    cₜ = Array{promote_type(eltype(generator), eltype(initialmatrix))}(
        undef, length(ts), size(initialmatrix)...
    )
    cₜ[1, :, :] .= initialmatrix

    # If L is the generator matrix and c₀ the initial correlation matrix,
    #   cₜ = transpose(exp(tL')) c₀ transpose(exp(-tL')) =
    #      = exp(t transpose(L')) c₀ exp(-t transpose(L')) =
    #      = exp(t conj(L)) c₀ exp(-t conj(L)).
    # Call V the matrix that diagonalises conj(L) as VDV^-1: then
    #   cₜ = V exp(tD) V^-1 c₀ V exp(-tD) V^-1.

    eigenvalues, V = eigen(conj(generator))
    D = Diagonal(eigenvalues)
    invV = if ishermitian(im * generator)
        V'
    else
        inv(V)
    end

    invVc₀V = invV * initialmatrix * V
    for j in eachindex(ts)[2:end]
        t = ts[j]
        cₜ[j, :, :] .= V * exp(t * D) * invVc₀V * exp(-t * D) * invV
    end
    return cₜ
end

function evolve_sf_correlation_matrix_step(ts::AbstractRange, generator, initialmatrix)
    cₜ = Array{promote_type(eltype(generator), eltype(initialmatrix))}(
        undef, length(ts), size(initialmatrix)...
    )
    # If L is the generator matrix and c₀ the initial correlation matrix,
    #   cₜ = transpose(exp(tL')) c₀ transpose(exp(-tL')) =
    #      = exp(t transpose(L')) c₀ exp(-t transpose(L')) =
    #      = exp(t conj(L)) c₀ exp(-t conj(L)) =
    #      = exp(t/N conj(L))^N c₀ exp(-t/N conj(L))^N

    # The `exp` function doesn't work on `BigFloat` matrices, but `eigen` does, and we can
    # exponentiate single `BigFloat`s, so we compute the exponential by first diagonalising
    # the generator.
    eigenvalues, V = eigen(conj(generator))
    # d, v = eigen(X)   ==>   v * Diagonal(d) * inv(v) ≈ X
    D = Diagonal(eigenvalues)
    invV = if ishermitian(im * generator)
        V'
    else
        inv(V)
    end

    dt = step(ts)
    exp_dtL = V * exp(dt * D) * invV
    inv_exp_dtL = V * exp(-dt * D) * invV

    cₜ[1, :, :] .= initialmatrix
    for j in 2:length(ts)
        cₜ[j, :, :] .= exp_dtL * cₜ[j - 1, :, :] * inv_exp_dtL
    end
    return cₜ
end

function evolve_sf_correlation_matrix_step(
    ts::AbstractRange, generator, initialmatrix, idxs...
)
    # Like `evolve_sf_correlation_matrix_step`, but don't save the whole matrix at each
    # time step, just some of its elements. Useful when memory is an issue.

    # The `exp` function doesn't work on `BigFloat` matrices, but `eigen` does, and we can
    # exponentiate single `BigFloat`s, so we compute the exponential by first diagonalising
    # the generator.
    eigenvalues, V = eigen(conj(generator))
    # d, v = eigen(X)   ==>   v * Diagonal(d) * inv(v) ≈ X
    D = Diagonal(eigenvalues)
    invV = if ishermitian(im * generator)
        V'
    else
        inv(V)
    end

    dt = step(ts)
    exp_dtL = V * exp(dt * D) * invV
    inv_exp_dtL = V * exp(-dt * D) * invV

    values = Array{promote_type(eltype(generator), eltype(initialmatrix))}(
        undef, length(ts), length(idxs)
    )
    # values[i,j] will be the matrix element cₜ[j] at time ts[i], i.e. `values[2.3, (4,5)]`.

    cₜ = initialmatrix
    for (i, idx) in enumerate(idxs)
        values[1, i] = getindex(cₜ, idx...)
    end

    for j in 2:length(ts)
        cₜ = exp_dtL * cₜ * inv_exp_dtL  # evolve matrix at t+dt
        for (i, idx) in enumerate(idxs)
            values[j, i] = getindex(cₜ, idx...)
        end
    end

    return values
end

function evolve_sf_correlation_matrix_step(ts, generator, initialmatrix)
    # More generic version of `evolve_sf_correlation_matrix_step` for non-uniform
    # time step.
    cₜ = Array{promote_type(eltype(generator), eltype(initialmatrix))}(
        undef, length(ts), size(initialmatrix)...
    )
    cₜ[1, :, :] .= initialmatrix

    for j in eachindex(ts)[2:end]
        dt = ts[j] - ts[j - 1]
        cₜ[j, :, :] .=
            exp(dt * conj(generator)) * cₜ[j - 1, :, :] * exp(-dt * conj(generator))
    end
    return cₜ
end

function evolve_sf_correlation_matrix_ode(tf, generator, initialmatrix)
    # If L is the generator matrix and c₀ the initial correlation matrix,
    #   cₜ = transpose(exp(tL')) c₀ transpose(exp(-tL')) =
    #      = exp(t transpose(L')) c₀ exp(-t transpose(L')) =
    #      = exp(t conj(L)) c₀ exp(-t conj(L)) =
    #      = exp(t/N conj(L))^N c₀ exp(-t/N conj(L))^N
    recL, imcL = reim(conj(generator))
    M = size(generator, 1)
    function lindblad_sf!(∂ₜcₜ, cₜ, p, t)
        ∂ₜcₜ[1:M, 1:M] .=
            recL * cₜ[1:M, 1:M] .- imcL * cₜ[1:M, (M + 1):end] .- cₜ[1:M, 1:M] * recL .+
            cₜ[1:M, (M + 1):end] * imcL
        ∂ₜcₜ[1:M, (M + 1):end] .=
            recL * cₜ[1:M, (M + 1):end] .+ imcL * cₜ[1:M, 1:M] .- cₜ[1:M, 1:M] * imcL .-
            cₜ[1:M, (M + 1):end] * recL
        return nothing
    end
    c₀ = zeros(M, 2M)
    c₀[1:M, 1:M] .= initialmatrix
    prob = ODEProblem(lindblad_sf!, c₀, (0, tf))
    return prob
end
