#!/usr/bin/julia

using DelimitedFiles, JSON
using TEDOPA

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
            # intervene to manually restrict the domain so that the part where the
            # transformed spectral densities are identically zero are removed.
            # It might happen that some of the domains become singleton, e.g. when
            # the original domain is [0, 1] and the associated chemical potential is 0.
            # This is fine for now, we will check this later.
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
    # TODO: check if there are gaps in the resulting merged domains.
    merge_domains(domains) = unique(sort(vcat(domains...)))
    merged_filled_domains = merge_domains(domains_filled)
    merged_empty_domains = merge_domains(domains_empty)

    function issingleton(domain)
        if !issorted(domain)
            error("Input domain is not sorted. Cannot check if it is empty.")
        end
        return minimum(domain) == maximum(domain)
    end

    # (We assume that the energies in the free environment Hamiltonians have already
    # been shifted with the chemical potentials. We won't do that here.)
    output_filename = replace(sd_info["filename"], ".json" => ".thermofield")

    if !issingleton(merged_empty_domains) && !issingleton(merged_filled_domains)
        (freqempty, coupempty, sysintempty) = chainmapcoefficients(
            merged_sdfempty,
            merged_empty_domains,
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
        )
        (freqfilled, coupfilled, sysintfilled) = chainmapcoefficients(
            merged_sdffilled,
            merged_filled_domains,
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
        )

        open(output_filename, "w") do output
            writedlm(output, ["coupempty" "coupfilled" "freqempty" "freqfilled"], ',')
            writedlm(
                output,
                [[sysintempty; coupempty] [sysintfilled; coupfilled] freqempty freqfilled],
                ',',
            )
        end
    elseif issingleton(merged_empty_domains)
        (freqfilled, coupfilled, sysintfilled) = chainmapcoefficients(
            merged_sdffilled,
            merged_filled_domains,
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
        )

        open(output_filename, "w") do output
            writedlm(output, ["coupfilled" "freqfilled"], ',')
            writedlm(output, [[sysintfilled; coupfilled] freqfilled], ',')
        end
    elseif issingleton(merged_filled_domains)
        (freqempty, coupempty, sysintempty) = chainmapcoefficients(
            merged_sdfempty,
            merged_empty_domains,
            chain_length - 1;
            Nquad=sd_info["PolyChaos_nquad"],
        )

        open(output_filename, "w") do output
            writedlm(output, ["coupempty" "freqempty"], ',')
            writedlm(output, [[sysintempty; coupempty] freqempty], ',')
        end
    else  # Both merged domains are singletons. There is no output.
        error("Both merged domains are empty. Please check the input spectral densities.")
    end

    return nothing
end
