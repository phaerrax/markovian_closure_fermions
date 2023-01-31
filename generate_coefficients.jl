#!/usr/bin/julia

using DataFrames, CSV, JSON, PolyChaos, SpecialFunctions
using PseudomodesTTEDOPA

disablegrifqtech()

indf(a::Real, b::Real, x::Real) = Int(a < x < b)

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

    # Do the same for its Fourier transform. Be careful not to reuse the same temporary
    # names for the variables: in some way, the function `sdf` above is not yet fixed,
    # so if we redefine `fn` and `tmp` to build the Fourier transform too the same `fn`
    # and `tmp` will end up in the previous `sdf`.
    fn_ft = sd_info["spectral_density_ft"]
    tmp_ft = eval(Meta.parse("(a, x) -> " * fn_ft))
    sdf_ft = x -> tmp_ft(sd_info["parameters"], x)

    ωmax = last(sd_info["domain"])
    chain_length = sd_info["number_of_oscillators"]

    # Compute the ftTEDOPA coefficients from the spectral density.
    T = sd_info["temperature"]
    μ = sd_info["chemical_potential"]
    if T == 0 && μ == 0
        # Straight TEDOPA.
        (Ω, κ, η) = chainmapcoefficients(
            sdf,
            sd_info["domain"],
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
            discretization=lanczos,
        )
    elseif μ == 0
        # T > 0, so we use the thermalized spectral density function.
        fT = ω -> thermalisedJ(sdf, ω, T)
        (Ω, κ, η) = chainmapcoefficients(
            fT,
            (-ωmax, 0, ωmax),
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
            discretization=lanczos,
        )
    elseif T == 0 # μ != 0
        fμ = ω -> indf(-ωmax + μ, μ, ω) * sdf(μ - ω) + indf(-μ, ωmax - μ, ω) * sdf(μ + ω)
        (Ω, κ, η) = chainmapcoefficients(
            fμ,
            (-ωmax, 0, ωmax),
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
            discretization=lanczos,
        )
    else # T != 0, μ != 0
        fTμ =
            ω ->
                0.5(1 + tanh(0.5ω / T)) *
                (indf(-ωmax + μ, μ, ω) * sdf(μ - ω) + indf(-μ, ωmax - μ, ω) * sdf(μ + ω))
        if ωmax > 2μ
            domainTμ = (-ωmax + μ, -μ, 0, μ, ωmax - μ)
        else
            domainTμ = (-μ, -ωmax + μ, 0, ωmax - μ, μ)
        end
        (Ω, κ, η) = chainmapcoefficients(
            fTμ,
            domainTμ,
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
            discretization=lanczos,
        )
    end

    dict = Dict(:loc => Ω, :int => [η; κ])
    table = DataFrame(dict)
    filename = replace(sd_info["filename"], ".json" => ".ttedopa.dat")
    CSV.write(filename, table)

    return nothing
end
