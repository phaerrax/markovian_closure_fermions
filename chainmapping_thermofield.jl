#!/usr/bin/julia

using DelimitedFiles, JSON
using TEDOPA

let
    file = ARGS[1]

    # We are assuming that the energies in the free environment Hamiltonians have already
    # been shifted with the chemical potentials.
    parameters = open(file, "r") do inputfile
        s = read(file, String)
        JSON.parse(s)
    end

    env_dicts = parameters["environment"]
    if env_dicts isa Dict
        envs = [env_dicts]
    else
        envs = env_dicts
    end
    chemical_potentials = [env["chemical_potential"] for env in envs]
    if any(!=(0), chemical_potentials)
        error(
            "The chainmapping_thermofield script can currently handle environments with " *
            "zero chemical potential only. Please edit the input file and shift all " *
            "spectral density functions accordingly so that μ = 0 for all of them.",
        )
    end

    coefficients = chainmapping_thermofield(file)

    output_filename = replace(file, ".json" => ".thermofield")
    open(output_filename, "w") do output
        @info "Output written on " * output_filename
        writedlm(output, ["coupempty" "coupfilled" "freqempty" "freqfilled"], ',')
        writedlm(
            output,
            [coefficients[:empty][:couplings] coefficients[:filled][:couplings] coefficients[:empty][:frequencies] coefficients[:filled][:frequencies]],
            ',',
        )
    end

    return nothing
end
