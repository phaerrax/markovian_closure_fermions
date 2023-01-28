#!/usr/bin/julia

# This code was written following the algorithm presented in
#
# │ Fernández Rodríguez, A., de Santiago Rodrigo, L., López Guillén, E. et al.
# │ “Coding Prony’s method in MATLAB and applying it to biomedical signal filtering”
# │ BMC Bioinformatics 19, 451 (2018)
# │ https://doi.org/10.1186/s12859-018-2473-y

using PolynomialRoots, LinearAlgebra

"""
    predictionmatrix(samples::AbstractVector, p::Int)

Return the "forward linear prediction matrix"

    ⎛ y[p]    y[p-1]  ⋯  y[1]   ⎞
    ⎜ y[p+1]  y[p]    ⋯  y[2]   ⎟
    ⎜  ⋮       ⋮      ⋱   ⋮     ⎟
    ⎝ y[N-1]  y[N-2]  ⋯  y[N-p] ⎠

with `y = samples`.
"""
function predictionmatrix(samples::AbstractVector, p::Int)
    N = length(samples)
    mat = zeros(eltype(samples), N - p, p)
    for rown in 1:(N - p), coln in 1:p
        mat[rown, coln] = samples[p + (rown - 1) - (coln - 1)]
    end
    return mat
end

"""
    observationvector(samples::AbstractVector, p::Int)

Construct the observation vector

     ⎛ y[p+1] ⎞
    -⎜ y[p+2] ⎟
     ⎜  ⋮     ⎟
     ⎝ y[N]   ⎠

with `y = samples`.
"""
function observationvector(samples::AbstractVector, p::Int)
    N = length(samples)
    v = zeros(eltype(samples), N - p)
    for i in 1:(N - p)
        v[i] = -samples[p + i]
    end
    return v
end

"""
    cproots(samples::AbstractVector, p::Int)

Find the coefficients ``aₖ``, ``k∈{1,…,p}`` of the characteristic polynomial
``ϕ(z) = ∏ₖ₌₁ᵖ (z-zₖ) = ∑ₖ₌₀ᵖ aₖzᵖ⁻ᵏ``.

They are the (approximate) solution to the linear system (`y = samples`)

    ⎛ y[p]    y[p-1]  ⋯  y[1]   ⎞ ⎛ a[1] ⎞    ⎛ y[p+1] ⎞  
    ⎜ y[p+1]  y[p]    ⋯  y[2]   ⎟ ⎜ a[2] ⎟ ≈ -⎜ y[p+2] ⎟ ;
    ⎜  ⋮       ⋮      ⋱   ⋮     ⎟ ⎜  ⋮   ⎟    ⎜ ⋮      ⎟ 
    ⎝ y[N-1]  y[N-2]  ⋯  y[N-p] ⎠ ⎝ a[p] ⎠    ⎝ y[N]   ⎠ 

`a[0]` is always 1.
"""
function cproots(samples::AbstractVector, p::Int)
    # A \ b returns the exact solution of Ax=b if A is square. For rectangular A
    # the result is the minimum-norm least squares solution.
    a = predictionmatrix(samples, p) \ observationvector(samples, p)
    # Look at the definition of ϕ: aₖ is the coefficient of the (p-k)-th power.
    # Since `roots` from PolynomialRoots requires that the roots from the
    # lowest coefficients to the highest, the list must be reversed.
    reverse!(a)
    # The last coefficient, a₀=1 (which multiplies zᵖ), is then added to the list.
    push!(a, one(eltype(a)))
    return roots(a)
end

function amplitudes(roots::AbstractVector, samples::AbstractVector)
    N = length(samples)
    p = length(roots)
    # Now that the roots zₖ, k∈{1,…,p} of ϕ are known, the constants hₖ can be
    # found from the equation
    #
    #   y[n] = ∑ₖ₌₁ᵖ hₖzₖⁿ⁻¹
    #
    # which can be written in matrix form as
    #
    #   ⎛ 1     1     ⋯  1     ⎞ ⎛ h₁ ⎞   ⎛ y[1] ⎞
    #   ⎜ z₁    z₂    ⋯  zₚ    ⎟ ⎜ h₂ ⎟   ⎜ y[2] ⎟
    #   ⎜ z₁²   z₂²   ⋯  zₚ²   ⎟ ⎜ h₃ ⎟ = ⎜ y[3] ⎟ .
    #   ⎜ ⋮     ⋮     ⋱  ⋮     ⎟ ⎜ ⋮  ⎟   ⎜  ⋮   ⎟
    #   ⎝ z₁ᴺ⁻¹ z₂ᴺ⁻¹ ⋯  zₚᴺ⁻¹ ⎠ ⎝ hₚ ⎠   ⎝ y[N] ⎠ 
    #
    Z = zeros(eltype(samples), N, p)
    for rown in 1:N, coln in 1:p
        Z[rown, coln] = roots[coln]^(rown - 1)
    end
    return Z \ samples
end

"""
    prony(f::Function, xstart::Real, xend::Real, xstep::Real, p::Int)

Apply Prony's algorithm to find the linear combination ``x ↦ ∑ⱼ₌₁ᵖ hⱼ exp(λⱼx)``
that best approximates `f` on [`xstart`, `xend`], sampled with step size
`xstep`.

Returns a Pair `(h,λ)` of lists that contain the coefficients.
"""
function prony(f::Function, xstart::Real, xend::Real, xstep::Real, p::Int)
    X = xstart:xstep:xend
    Y = f.(X)
    if 2p > length(X)
        error("2p must be smaller than the number of data points.")
    end
    z = cproots(Y, p)
    h = amplitudes(z, Y)
    # We have successfully approximated Y = {yₙ}ₙ₌₁ᴺ as
    #   
    #   yₙ = ∑ₖ₌₁ᵖ hₖ zₖⁿ⁻¹ (n ≥ 1)
    #
    # where n ∈ eachindex(X). If we define zₖ := exp(λₖ) then
    #   
    #   yₙ = ∑ₖ₌₁ᵖ hₖ exp(λₖ(n-1)).
    #
    λ = log.(z)
    # Now n must be traced back to the values in X, which is assumed to be a
    # list of equidistant, ordered, values xₙ. If τ is the step size of X, i.e.
    #
    #   X = { nτ : n ∈ 𝐍 and n < ̄n }
    #   
    # then n-1 = τ⁻¹ τ(n-1) = xₙ/τ, therefore
    #
    #   y(xₙ) = ∑ₖ₌₁ᵖ hₖ exp(λₖxₙ/τ).
    #
    # The coefficients we want are then hₖ and λₖ/τ. It's okay if one of them
    # depends on τ: after all this parameter determines which values yₙ are
    # picked up by the algorithm. Hopefully the final result doesn't depend
    # on the particular choice of τ (it seems like it doesn't...).
    return h, λ ./ xstep
end
