struct ModeChain
    range
    frequencies
    couplings
    function ModeChain(input_range, input_frequencies, input_couplings)
        @assert allequal([
            length(input_range), length(input_frequencies), length(input_couplings) + 1
        ])
        return new(input_range, input_frequencies, input_couplings)
    end
end

function spinchain(::SiteType"Fermion", c::ModeChain)
    ad_h = OpSum()
    for (n, f) in zip(c.range, c.frequencies)
        ad_h += f, "n", n
    end
    for (n1, n2, g) in zip(c.range[1:(end - 1)], c.range[2:end], c.couplings)
        ad_h += g, "cdag", n1, "c", n2
        ad_h += g, "cdag", n2, "c", n1
    end
    return ad_h
end

function spinchain(::SiteType"vFermion", c::ModeChain)
    ℓ = OpSum()
    for (n, f) in zip(c.range, c.frequencies)
        ℓ += f * gkslcommutator("N", n)
    end
    for (n1, n2, g) in zip(c.range[1:(end - 1)], c.range[2:end], c.couplings)
        # Reorder `n1` and `n2`, otherwise `jwstring` doesn't put in the string factors
        # if `n1 > n2`. The exchange interaction operator is symmetric in `n1` and `n2`
        # so this reordering doesn't affect the result.
        n1, n2 = minmax(n1, n2)
        jws = jwstring(; start=n1, stop=n2)
        ℓ +=
            g * (
                gkslcommutator("A†", n1, jws..., "A", n2) +
                gkslcommutator("A", n1, jws..., "A†", n2)
            )
    end
    return ℓ
end

"""
    reverse(c::ModeChain)

Invert the direction of the mode chain `c`, by reversing its range, its frequencies array
and its coupling constants array.

# Example

```julia-repl
julia> c = ModeChain([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'], ['w', 'x', 'y', 'z']);

julia> reverse(c)
ModeChain([5, 4, 3, 2, 1], ['e', 'd', 'c', 'b', 'a'], ['z', 'y', 'x', 'w'])
```
"""
Base.reverse(c::ModeChain) =
    ModeChain(reverse(c.range), reverse(c.frequencies), reverse(c.couplings))

"""
    join(c1::ModeChain, c2::ModeChain, c1c2coupling)

Return a new `ModeChain` made by joining `c1` and `c2`, with `c1c2coupling` as the coupling
constant between the last site of `c1` and the first site of `c2`.
The ranges of the two chains do not have to be adjacent, or ordered in some way, but they
cannot overlap.

# Example

```julia-repl
julia> r1 = 2:2:10; r2 = 3:2:10;

julia> c1 = ModeChain(r1, fill("f1", length(r1)), fill("g1", length(r1) - 1));

julia> c2 = ModeChain(r2, fill("f2", length(r2)), fill("g2", length(r2) - 1));

julia> join(c1, c2, "r")
ModeChain(
    [2, 4, 6, 8, 10, 3, 5, 7, 9],
    ["f1", "f1", "f1", "f1", "f1", "f2", "f2", "f2", "f2"],
    ["g1", "g1", "g1", "g1", "r", "g2", "g2", "g2"],
)
```
"""
function Base.join(c1::ModeChain, c2::ModeChain, c1c2coupling)
    if isempty(intersect(c1.range, c2.range))
        return ModeChain(
            [c1.range; c2.range],
            [c1.frequencies; c2.frequencies],
            [c1.couplings; c1c2coupling; c2.couplings],
        )
    else
        ArgumentError("Ranges overlap")
    end
end

Base.length(c::ModeChain) = length(c.range)
Base.iterate(c::ModeChain) = iterate(c.range)
Base.iterate(c::ModeChain, i::Int) = iterate(c.range, i)

"""
    first(c::ModeChain, n::Integer)

Get the the mode chain `c` truncated to its first `n` elements.
"""
Base.first(c::ModeChain, n::Integer) =
    ModeChain(first(c.range, n), first(c.frequencies, n), first(c.couplings, n - 1))

"""
    last(c::ModeChain, n::Integer)

Get the the mode chain `c` truncated to its last `n` elements.
"""
Base.last(c::ModeChain, n::Integer) =
    ModeChain(last(c.range, n), last(c.frequencies, n), last(c.couplings, n - 1))

"""
    markovianclosure(chain::ModeChain, nclosure, nenvironment)

Replace the ModeChain `chain` with a truncated chain of `nenvironment` elements plus a
Markovian closure made of `nclosure` pseudomodes.
The asymptotic frequency and coupling coefficients are determined automatically from the
average of the chain cofficients from `nenvironment+1` to the end, but they can also be
given manually with the keyword arguments `asymptoticfrequency` and `asymptoticcoupling`.
"""
function MarkovianClosure.markovianclosure(
    chain::ModeChain,
    nclosure,
    nenvironment;
    asymptoticfrequency=nothing,
    asymptoticcoupling=nothing,
)
    if isnothing(asymptoticfrequency)
        asymptoticfrequency = mean(chain.frequencies[(nenvironment + 1):end])
    end
    if isnothing(asymptoticcoupling)
        asymptoticcoupling = mean(chain.couplings[(nenvironment + 1):end])
    end

    truncated_envchain = first(chain, nenvironment)
    mc = markovianclosure_parameters(asymptoticfrequency, asymptoticcoupling, nclosure)

    return truncated_envchain, mc
end
