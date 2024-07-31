#!/usr/bin/julia

using DelimitedFiles
using TEDOPA

let
    # Load information on the spectral density from file.
    file = ARGS[1]
    coefficients = chainmapping_tftedopa(file)

    open(replace(file, ".json" => ".tedopa"), "w") do output
        writedlm(output, ["couplings" "frequencies"], ',')
        writedlm(output, [coefficients[:couplings] coefficients[:frequencies]], ',')
    end

    return nothing
end
