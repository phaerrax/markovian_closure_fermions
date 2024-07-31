#!/usr/bin/julia

using DelimitedFiles
using TEDOPA

let
    # Load information on the spectral density from file.
    file = ARGS[1]
    coefficients = chainmapping_tftedopa(file)

    output_filename = replace(file, ".json" => ".tedopa")
    open(output_filename, "w") do output
        @info "Output written on " * output_filename
        writedlm(output, ["couplings" "frequencies"], ',')
        writedlm(output, [coefficients[:couplings] coefficients[:frequencies]], ',')
    end

    return nothing
end
