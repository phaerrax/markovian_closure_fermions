interleave(v...) = collect(Iterators.flatten(zip(v...)))

function enlargelinks_delta(v::MPS, new_d)
    N = length(v)
    if N == 1
        v_overlap = 1
        @debug "The length of the MPS is 1. There are no bonds to enlarge."
        return v, v_overlap
    end

    v_ext = copy(v)
    if hasqns(v[1])
        @warn "This function currently doesn't work when QNs are involved."
        #=
           The `hasqns` part comes from the constructor method of an MPS with QNs.
           The QN index structure of an MPS is as follows:

                                      s[n]                        s[n+1]

                            │                            │
                            │      l[n]                  │     l[n+1]
                          ╭───╮    <In>                ╭───╮   <In>
           ───────────────│ n │───────╮ ╭──────────────│n+1│─────────
           dag(l[n-1])    ╰───╯       │ │ dag(l[n])    ╰───╯
           <Out>                      │ │ <Out>
                                      │ │
                            ╭─────────╯ ╰─────────╮
                            │                     │
                       ┌╶╶╶╶╶╶╶╶╶╶╶╶╶┐      ┌╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶┐
                       ╎delta(       ╎      ╎delta(          ╎
                       ╎  dag(l[n]), ╎      ╎  l[n],         ╎
                       ╎  new_index  ╎      ╎  dag(new_index)╎
                       ╎)            ╎      ╎)               ╎
                       └╶╶╶╶╶╶╶╶╶╶╶╶╶┘      └╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶┘
                            │                     │

        =#
        # We need now to calculate the flux of the QNs in the MPS. Note that in the original
        # MPS constructor, where this code comes from, the flux is calculated from the
        # states that make up the product-state MPS, not from the MPS itself (that obviously
        # doesn't yet exist there). We need to recreate those states starting from the MPS
        # `v`: we cannot use just `v` because it contains link indices too, and they mess
        # up the calculation of the flux.
        # If the MPS is a product state, then we can recover the states by contracting the
        # link indices on each site: since they are one-dimensional, we contract them with
        # an ITensor made from `onehot(... => 1)`, and the link index disappears without
        # affecting the rest.
        if any(linkdims(v_ext) .!= 1)
            error(
                "MPS is not a trivial product state. Enlarging non-product-states is " *
                "currently not supported.",
            )
        end
        states = Vector{ITensor}(undef, N)
        states[1] = v_ext[1] * dag(onehot(linkind(v_ext, 1) => 1))
        for n in 2:(N - 1)
            states[n] =
                v_ext[n] *
                onehot(linkind(v_ext, n - 1) => 1) *
                onehot(dag(linkind(v_ext, n)) => 1)
        end
        states[N] = v_ext[N] * onehot(linkind(v_ext, N - 1) => 1)

        enlarged_links = Vector{ITensors.QNIndex}(undef, N - 1)
        lflux = sum(flux, states[1:(end - 1)])
        for bond in (N - 1):-1:1
            enlarged_links[bond] = dag(
                Index(lflux => new_d; tags=tags(commonind(v_ext[bond], v_ext[bond + 1])))
            )
            lflux -= flux(states[bond])
        end
    else
        enlarged_links = [
            Index(new_d; tags=tags(commonind(v_ext[bond], v_ext[bond + 1]))) for
            bond in 1:(N - 1)
        ]
    end

    for bond in 1:(N - 1)
        bond_index = commonind(v_ext[bond], v_ext[bond + 1])
        # Remember to `dag` some indices so that the directions of the QN match (see above)
        v_ext[bond] = v_ext[bond] * delta(dag(bond_index), enlarged_links[bond])
        v_ext[bond + 1] = v_ext[bond + 1] * delta(bond_index, dag(enlarged_links[bond]))
    end

    v_overlap = dot(v, v_ext)
    @debug "Overlap ⟨original|extended⟩: $v_overlap"
    return v_ext, v_overlap
end

nullfile() = "/dev/null"

"""
    pack!(
        outputfilename;
        argsdict,
        expvals_file,
        bonddimensions_file,
        walltime_file,
        finalstate=nothing,
    )

Gather all simulation input and output in a single HDF5 file `outputfilename`.
All input parameters used for the simulation, except for the name of the JSON input file
itself (if provided) will be in the output file together with the expectation values of the
selected observables, the bond dimensions of the evolved MPS and the wall-clock time spent
computing each step, for each step of the time evolution.

Optionally store also the final state of the evolution by passing it to the function
under the `finalstate` keyword argument.

The `expvals_file`, `bonddimensions_file` and `walltime_file` files will be _deleted_ after
their contents are transferred to the HDF5 file.
"""
function pack!(
    outputfilename;
    argsdict,
    expvals_file,
    bonddimensions_file,
    walltime_file,
    finalstate=nothing,
)
    # We remove the "observable" entry from the dictionary since we don't need it in the
    # output, it's already in the headers of the simulation results.
    # Same for "input_parameters", since its contents are what we're actually writing in
    # the HDF5 file.
    dict = deepcopy(argsdict)
    delete!(dict, "observables")
    delete!(dict, "input_parameters")
    delete!(dict, "name")

    h5open(outputfilename, "w") do hf
        for (k, v) in dict
            write(hf, k, v)
        end
        measurements = CSV.File(expvals_file)
        for col in propertynames(measurements)
            # If Julia is run with multithreading active and the CSV file is large
            # (> 5_000 cells), `CSV.read` will use multiple threads to read the file, and
            # the columns will be read as `SentinelArrays.ChainedVector{T, Vector{T}}`
            # instead of `Vector{T}`. We use `collect` to transform the former into the
            # latter, in case this happens, since `HDF5.write` accepts only simple vectors.
            write(hf, string("simulation_results/", col), collect(measurements[col]))
        end

        # Pack the bond dimensions in a single Matrix, where the i-th columns is the link
        # between site i and i+1.
        # (`Tables.matrix` already converts `SentinelArrays.ChainedVector{T, Vector{T}}`s
        # into normal matrices.)
        bonddims = Matrix{Int}(Tables.matrix(CSV.File(bonddimensions_file; drop=[:time])))
        write(hf, "bond_dimensions", bonddims)

        # Read the simulation time per step in a Vector. There's only one column in the
        # file.
        walltime = CSV.File(walltime_file)
        write(hf, "simulation_wall_time", collect(walltime[propertynames(walltime)[1]]))

        if !isnothing(finalstate)
            write(hf, "final_state", finalstate)
        end
    end

    @info "Output written on $outputfilename"
    rm(expvals_file)
    rm(bonddimensions_file)
    rm(walltime_file)

    return nothing
end

function simulation_files_info(;
    measurements_file=nothing, bonddims_file=nothing, simtime_file=nothing
)
    str = "You can follow the time evolution step by step in the following files:"
    if !(isnothing(measurements_file) || measurements_file == nullfile())
        str *= "\n$measurements_file\t for the expectation values"
    end
    if !(isnothing(bonddims_file) || bonddims_file == nullfile())
        str *= "\n$bonddims_file\t for the bond dimensions of the evolved MPS"
    end
    if !(isnothing(simtime_file) || simtime_file == nullfile())
        str *= "\n$simtime_file\t for the wall-clock time spent computing each step"
    end
    @info str
end
