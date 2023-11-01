#!/usr/bin/julia

using DelimitedFiles, JSON
using PseudomodesTTEDOPA

let
    # Load information on the spectral density from file.
    file = ARGS[1]

    p = open(file) do inp
        s = read(inp, String)
        return JSON.parse(s)
    end
    sd_info = merge(p, Dict("filename" => file))

    # Parse the spectral density function given in the info file.
    sdfs = []
    for d in sd_info["environments"]
        tmp = eval(Meta.parse("(a, x) -> " * d["spectral_density"]))
        push!(sdfs, x -> tmp(d["parameters"], x))
    end
    Ts = [d["temperature"] for d in sd_info["environments"]]
    μs = [d["chemical_potential"] for d in sd_info["environments"]]
    domains = [d["domain"] for d in sd_info["environments"]]

    chain_length = sd_info["number_of_oscillators"]

    domains_empty = []
    domains_filled = []
    for (T, μ, domain) in zip(Ts, μs, domains)
        if T == 0
            # Even though Julia is able to handle T = 0 in the formulae, we still need to
            # intervene to manually restrict the domain so that the part where the transformed
            # spectral densities are identically zero are removed.
            push!(domains_empty, [μ, filter(>(μ), domain)...])
            push!(domains_filled, [filter(<(μ), domain)..., μ])
        else
            push!(domains_empty, domain)
            push!(domains_filled, domain)
        end
    end

    n(β, μ, ω) = (exp(β * (ω - μ)) + 1)^(-1)

    function merged_sdfempty(ω)
        return sum([(1 - n(1 / T, μ, ω)) * f(ω) for (f, T, μ) in zip(sdfs, Ts, μs)])
    end
    function merged_sdffilled(ω)
        return sum([n(1 / T, μ, ω) * f(ω) for (f, T, μ) in zip(sdfs, Ts, μs)])
    end

    # To merge the domains: concatenate, sort, then remove duplicates
    merge_domains(domains) = unique(sort(vcat(domains...)))

    (freqempty, coupempty, sysintempty) = chainmapcoefficients(
        merged_sdfempty,
        merge_domains(domains_empty),
        chain_length - 1;
        Nquad=sd_info["PolyChaos_nquad"],
    )
    (freqfilled, coupfilled, sysintfilled) = chainmapcoefficients(
        merged_sdffilled,
        merge_domains(domains_filled),
        chain_length - 1;
        Nquad=sd_info["PolyChaos_nquad"],
    )
    # We assume that the energies in the free environment Hamiltonians have already
    # been shifted with the chemical potentials. We won't do that here.

    open(replace(sd_info["filename"], ".json" => ".thermofield"), "w") do output
        writedlm(output, ["coupempty" "coupfilled" "freqempty" "freqfilled"], ',')
        writedlm(
            output,
            [[sysintempty; coupempty] [sysintfilled; coupfilled] freqempty freqfilled],
            ',',
        )
    end

    return nothing
end
