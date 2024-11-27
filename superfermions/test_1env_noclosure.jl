using ITensors, ITensorMPS, TimeEvoVecMPS, ArgParse, MKL

function enlargelinks_delta(v::MPS, new_d)
    @warn "This function currently doesn't seem to work when QNs are involved."
    N = length(v)
    if N == 1
        v_overlap = 1
        @debug "The length of the MPS is 1. There are no bonds to enlarge."
        return v, v_overlap
    end

    v_ext = copy(v)
    if hasqns(v[1])
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
                "MPS is not a triv_extial product state. Enlarging non-product-states is " *
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

"""
    enlargelinks(v, dims; ref_state)

Increase the bond dimensions of the MPS `v` to `dims` by adding an all-zero MPS to it,
with a direct sum (it returns a new MPS).
In order to obtain the all-zero MPS, a random MPS with the given bond dimensions is
computed, so that the correct link structure is maintained; it is then multiplied by zero.

If the MPS has some conserved quantum numbers, then a product state must also be supplied
as the keyword argument `ref_state`, because the random MPS is obtained by randomising this
initial state (which also determines the total QN of the resulting random MPS).
This argument may take the form of one of the many available ways to build an MPS from the
`MPS(sites, ...)` function, such as a string, an array of strings, or a function.

Note that this function will _always_ increase the dimension of all links by 1 at least.

# Examples

```julia-repl
julia> N = 20;

julia> s = siteinds("Fermion", N; conserve_nfparity=true);

julia> v = MPS(s, "Emp");

julia> enlargelinks(v, 10; ref_state="Emp");

julia> enlargelinks(v, 10; ref_state=n -> isodd(n) ? "Occ" : "Emp");

julia> enlargelinks(v, 10; ref_state=[n ≤ N/2 ? "Occ" : "Emp" for n in 1:N]);
```
"""
function enlargelinks(v, dims::Vector{<:Integer}; ref_state=nothing)
    diff_linkdims = max.(dims .- linkdims(v), 1)
    x = if hasqns(first(v))
        if isnothing(ref_state)
            error("Initial state required to use enlargelinks with QNs")
        else
            random_mps(siteinds(v), ref_state; linkdims=diff_linkdims)
        end
    else
        random_mps(siteinds(v); linkdims=diff_linkdims)
    end
    orthogonalize!(x, 1)
    return add(orthogonalize(v, 1), 0 * x; alg="directsum")
end

function enlargelinks(v, dims::Integer; kwargs...)
    return enlargelinks(v, fill(dims, length(v) - 1); kwargs...)
end

nullfile() = "/dev/null"

struct ModeChain
    range
    frequencies
    couplings
    function ModeChain(input_range, input_frequencies, input_couplings)
        @assert allequal([
            length(input_range), length(input_frequencies), length(input_couplings) + 1
        ])
        return new(input_range, input_frequencies, input_couplings)
    end
end

function spinchain(::SiteType"Fermion", c::ModeChain)
    ad_h = OpSum()
    for (n, f) in zip(c.range, c.frequencies)
        ad_h += f, "n", n
    end
    for (n1, n2, g) in zip(c.range[1:(end - 1)], c.range[2:end], c.couplings)
        ad_h += g, "cdag", n1, "c", n2
        ad_h += g, "cdag", n2, "c", n1
    end
    return ad_h
end

function join(c1::ModeChain, c2::ModeChain, c1c2coupling)
    if first(c1.range) ≤ last(c2.range) && first(c2.range) ≤ last(c1.range)
        error("The ranges of the given ModeChains overlap.")
    elseif first(c1.range) < first(c2.range)  # find out which chain is on the left
        return ModeChain(
            [c1.range; c2.range],
            [c1.frequencies; c2.frequencies],
            [c1.couplings; c1c2coupling; c2.couplings],
        )
    elseif first(c2.range) < first(c1.range)
        return ModeChain(
            [c2.range; c1.range],
            [c2.frequencies; c1.frequencies],
            [c2.couplings; c1c2coupling; c1.couplings],
        )
    else
        error("?")
    end
end

function simulation(;
    nsystem,
    nenvironment,
    dt,
    tmax,
    maxbonddim,
    io_file=nullfile(),
    io_ranks=nullfile(),
    io_times=nullfile(),
)
    sites = siteinds("Fermion", 2nsystem + 2nenvironment; conserve_nfparity=true)
    system = ModeChain(range(; start=1, step=2, length=nsystem), [1], [])
    altsystem = ModeChain(range(; start=2, step=2, length=nsystem), [1], [])

    sysenvcoupling = 1
    environment = ModeChain(
        range(; start=2nsystem + 1, step=2, length=nenvironment),
        fill(1, nenvironment),
        fill(1, nenvironment - 1),
    )
    altenvironment = ModeChain(
        range(; start=2nsystem + 2, step=2, length=nenvironment),
        fill(1, nenvironment),
        fill(1, nenvironment - 1),
    )

    initstate = MPS(sites, n -> n ≤ 2 ? "Occ" : "Emp")

    ad_h =
        spinchain(SiteType("Fermion"), join(system, environment, sysenvcoupling)) -
        spinchain(SiteType("Fermion"), join(altsystem, altenvironment, sysenvcoupling))

    L = MPO(-im * ad_h, sites)

    operators = [
        LocalOperator(Dict(1 => "n"))
        LocalOperator(Dict(2 => "n"))
        LocalOperator(Dict(3 => "n"))
        LocalOperator(Dict(4 => "n"))
        LocalOperator(Dict(2nsystem + 2nenvironment - 1 => "n"))
        LocalOperator(Dict(2nsystem + 2nenvironment => "n"))
        LocalOperator(Dict(1 => "cdag", 3 => "c"))
        LocalOperator(Dict(3 => "cdag", 1 => "c"))
        LocalOperator(Dict(2 => "c", 4 => "cdag"))
    ]
    cb = SuperfermionCallback(operators, sites, dt)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions. We have three methods available.

    # 1. `expand` from ITensorsMPS, which implements the global subspace expansion
    #    algorithm. It returns an enlarged MPS, but it is not large enough, i.e. sometimes
    #    its bond dimensions are still 1 on some links, preventing a correct TDVP run.
    # state_t = expand(initstate, L; alg="global_krylov", krylovdim=maxbonddim-1)

    # 2. Old `growMPS` method from TimeEvoVecMPS, contracting each link index with a delta
    #    ITensor with bigger dimensions. Revisited so that it takes QNs into account, but
    #    there must be a bug somewhere because the TDVP evolution does not give the
    #    expected results.
    # state_t, _ = enlarge(initstate, maxbonddim)

    # 3. New method, that sums an all-zero MPS with the necessary bond dimensions to
    #    `initstate`. The sum is a direct sum, so that no truncation is performed. It seems
    #    to work, but the resulting structure of the link indices is still a mystery.
    state_t = enlargelinks(initstate, maxbonddim; ref_state=n -> n ≤ 2 ? "Occ" : "Emp")

    tdvp1vec!(
        state_t,
        L,
        dt,
        tmax,
        sites;
        hermitian=false,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=io_file,
        io_ranks=io_ranks,
        io_times=io_times,
        superfermions=true,
    )

    return nothing
end

function parsecommandline()
    @info "Reading parameters from command line"
    s = ArgParseSettings()
    @add_arg_table s begin
        "--system_sites", "--ns"
        help = "Number of system sites"
        arg_type = Int
        required = true
        "--environment_sites", "--ne"
        help = "Number of environment sites"
        arg_type = Int
        required = true
        "--time_step", "--dt"
        help = "Time step of the evolution"
        arg_type = Float64
        required = true
        "--max_time", "--maxt"
        help = "Total physical time of the evolution"
        arg_type = Float64
        required = true
        "--bond_dimension", "--bdim"
        help = "Bond dimension of the state MPS"
        arg_type = Int
        "--output_file", "-o"
        help = "Path of output file"
        arg_type = String
        required = true
    end

    # Convert keys from String to Symbol and remove the ones whose value is `nothing`.
    parsedargs = Dict()
    for (k, v) in parse_args(s)
        isnothing(v) || push!(parsedargs, Symbol(k) => v)
    end
    return parsedargs
end

function main()
    parsedargs = parsecommandline()
    simulation(;
        nsystem=parsedargs[:system_sites],
        nenvironment=parsedargs[:environment_sites],
        dt=parsedargs[:time_step],
        tmax=parsedargs[:max_time],
        maxbonddim=parsedargs[:bond_dimension],
        io_file=parsedargs[:output_file],
    )

    return nothing
end

main()
