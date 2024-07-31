#!/usr/bin/julia

using DelimitedFiles
using TEDOPA

let
    file = ARGS[1]

    # We are assuming that the energies in the free environment Hamiltonians have already
    # been shifted with the chemical potentials.
    parameters = open(filename, "r") do inputfile
        s = read(file, String)
        JSON.parse(s)
    end
    if parameters["chemical_potential"] != 0
        error(
            "The chainmapping_thermofield script can currently handle environments with " *
            "zero chemical potential only. Please edit the input file and shift all " *
            "spectral density functions accordingly so that μ = 0 for all of them."
        )
    end

    coefficients = chainmapping_thermofield(file)

    output_filename = replace(file, ".json" => ".thermofield")

    open(output_filename, "w") do output
        writedlm(output, ["coupempty" "coupfilled" "freqempty" "freqfilled"], ',')
        writedlm(
            output,
            [coefficients[:empty][:couplings] coefficients[:filled][:couplings] coefficients[:empty][:frequencies] coefficients[:filled][:frequencies]],
            ',',
        )
    end

    return nothing
end
