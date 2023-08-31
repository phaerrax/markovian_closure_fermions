#!/usr/bin/julia

using DelimitedFiles, JSON
using PseudomodesTTEDOPA

disablegrifqtech()

let
    # Load information on the spectral density from file.
    file = ARGS[1]
    local p
    open(file) do inp
        s = read(inp, String)
        p = JSON.parse(s)
    end
    sd_info = merge(p, Dict("filename" => file))

    # Parse the spectral density function given in the info file.
    fn = sd_info["spectral_density"]
    tmp = eval(Meta.parse("(a, x) -> " * fn))
    sdf = x -> tmp(sd_info["parameters"], x)
    T = sd_info["temperature"]
    μ = sd_info["chemical_potential"]
    ωmax = last(sd_info["domain"])

    # Shift the spectral density function so that it is centred around zero.
    sdf′(x) = sdf(x + 0.5ωmax)
    domain′ = sd_info["domain"] .- 0.5ωmax
    μ′ = μ - 0.5ωmax

    chain_length = sd_info["number_of_oscillators"]

    # From the original spectral density J: Ω -> [0, +∞) we create two new environments
    # with spectral densities J₊: Ω -> [0, +∞) and J₋: Ω -> [0, +∞) given by
    #   J₋(ω) = (1 - n(ω)) J(ω),
    #   J₊(ω) = n(ω) J(ω)
    # defined on Ω.
    # We use the standard TEDOPA chain mapping to transform these two environments into
    # two linear chains: J₊ will be associated to an initially empty chain, and J₋ to an
    # initially filled chain.

    if T == 0
        # Even though Julia is able to handle T = 0 in the formulae, we still need to
        # intervene to manually restrict the domain so that the part where the transformed
        # spectral densities are identically zero are removed.
        domainempty = (μ′, filter(>(μ′), domain′)...)
        domainfilled = (filter(<(μ′), domain′)..., μ′)
    else
        domainempty = domain′
        domainfilled = domain′
    end

    n(β, μ, ω) = (exp(β * (ω - μ)) + 1)^(-1)
    sdfempty = ω -> (1 - n(1 / T, μ′, ω)) * sdf′(ω)
    sdffilled = ω -> n(1 / T, μ′, ω) * sdf′(ω)

    (freqempty, coupempty, sysintempty) = chainmapcoefficients(
        sdfempty, domainempty, chain_length - 1; Nquad=sd_info["PolyChaos_nquad"]
    )
    (freqfilled, coupfilled, sysintfilled) = chainmapcoefficients(
        sdffilled, domainfilled, chain_length - 1; Nquad=sd_info["PolyChaos_nquad"]
    )
    freqfilled .-= μ′
    freqempty .-= μ′

    open(replace(sd_info["filename"], ".json" => ".thermofield0"), "w") do output
        writedlm(output, ["coupempty" "coupfilled" "freqempty" "freqfilled"], ',')
        writedlm(
            output,
            [[sysintempty; coupempty] [sysintfilled; coupfilled] freqempty freqfilled],
            ',',
        )
    end

    return nothing
end
