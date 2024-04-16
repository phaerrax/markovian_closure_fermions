#!/usr/bin/julia

using DelimitedFiles, JSON
using Plots, TEDOPA

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

    sdf_plot = plot()
    xstep(domain; step=0.001) = first(domain):step:last(domain)
    for (dom, f) in zip(domains, sdfs)
        plot!(sdf_plot, xstep(dom), f.(xstep(dom)); linestyle=:dash)
    end

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

    # This function removes from the domain those values for which the function is below
    # the given threshold. We use it to fix the domain of functions that are "almost zero"
    # in some region, and would cause trouble in the convergence of the coefficients if
    # left unchecked.
    # We deliberately take the interval between the first and last point of the filtered
    # domain, since we cannot admit gaps in the domain of a Szegő-class function, so
    # we might as well leave in the result everything in between. If the resulting domain
    # is such that the function is zero inside it, then the function is "pathological" and
    # should have never been used for the TEDOPA algorithm in the first place.
    function exclude_almost_zero(domain, f; threshold=1e-8)
        list = filter(x -> f(x) > threshold, xstep(domain))
        min = first(list)
        max = last(list)
        return [min; filter(x -> min < x < max, domain); max]
    end

    # To merge the domains: concatenate, sort, then remove duplicates
    # TODO: check if there are gaps in the resulting merged domains.
    merge_domains(domains) = unique(sort(vcat(domains...)))
    merged_filled_domains = exclude_almost_zero(
        merge_domains(domains_filled), merged_sdffilled
    )
    merged_empty_domains = exclude_almost_zero(
        merge_domains(domains_empty), merged_sdfempty
    )

    function issingleton(domain)
        if !issorted(domain)
            error("Input domain is not sorted. Cannot check if it is empty.")
        end
        return minimum(domain) == maximum(domain)
    end

    # (We assume that the energies in the free environment Hamiltonians have already
    # been shifted with the chemical potentials. We won't do that here.)
    output_filename = replace(sd_info["filename"], ".json" => ".thermofield")
    plotfig_filename = replace(sd_info["filename"], ".json" => "_plots.png")

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
        plot!(
            sdf_plot,
            xstep(merged_filled_domains),
            merged_sdffilled.(xstep(merged_filled_domains)),
        )
        plot!(
            sdf_plot,
            xstep(merged_empty_domains),
            merged_sdfempty.(xstep(merged_empty_domains)),
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
        plot!(
            sdf_plot,
            xstep(merged_filled_domains),
            merged_sdffilled.(xstep(merged_filled_domains)),
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
        plot!(
            sdf_plot,
            xstep(merged_empty_domains),
            merged_sdfempty.(xstep(merged_empty_domains)),
        )

        open(output_filename, "w") do output
            writedlm(output, ["coupempty" "freqempty"], ',')
            writedlm(output, [[sysintempty; coupempty] freqempty], ',')
        end
    else  # Both merged domains are singletons. There is no output.
        error("Both merged domains are empty. Please check the input spectral densities.")
    end

    savefig(sdf_plot, plotfig_filename)

    return nothing
end
