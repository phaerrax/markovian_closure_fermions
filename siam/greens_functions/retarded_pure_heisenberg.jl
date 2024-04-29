using ITensors
using ITensors.HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using TimeEvoVecMPS

let
    parameters = load_pars(ARGS[1])

    # Load initial state from disk
    initstate_file = get(parameters, "initial_state_file", nothing)
    psi_0 = h5open(initstate_file, "r") do file
        return read(file, parameters["initial_state_label"], MPS)
    end
    sites = siteinds(psi_0)
    vsites = siteinds("vFermion", length(sites))

    # Construct the operator a_1†|ψ⟩⟨ψ| from the initial pure state ψ
    rho_0 = outer(apply(op("a", sites, 1), psi_0), psi_0')
    # Vectorize it and convert it into an MPS. The "combiner" index merges the
    # two matrix indices in a single one, effectively transforming the matrix
    # into a vector. Our vectorized fermion site type works in the Gell-Mann
    # basis though... we need to implement a change of basis on each site.
    #
    # If t = [a b; c d] with index `s` then
    #   t * combiner(s', s) = [a, c, b, d]  (stacks columns)
    #   t * combiner(s, s') = [a, b, c, d]  (stacks rows)
    #
    # Let's stack by rows, and define
    #   e[1] = [1 0; 0 0]
    #   e[2] = [0 1; 0 0]
    #   e[3] = [0 0; 1 0]
    #   e[4] = [0 0; 0 1]
    # so that [a b; c d] = a*e[1] + b*e[2] + c*e[3] + d*e[4].
    # The 2d Gell-Mann basis (our version of it) is
    #   f[1] = 1/sqrt(2) [1 0; 0 -1]
    #   f[2] = 1/sqrt(2) [0 1; 1 0]
    #   f[3] = 1/sqrt(2) [0 -i; i 0]
    #   f[4] = 1/sqrt(2) [1 0; 0 1]
    # and with these definitions we get
    #   e[i] = sum([U[i,j] * f[j] for j in 1:4])
    # with
    #        1  ⎛  1  0  0  1 ⎞
    #   U = ⸺   ⎜  0  1  i  0 ⎟
    #       √2  ⎜  0  1 -i  0 ⎟
    #           ⎝ -1  0  0  1 ⎠
    # Note that coordinate vectors change basis with the transpose of this matrix,
    # so we need to multiply each blocks of the MPS by U on the _left_.
    # We achieve this by defining the tensor with a specific order of the indices.
    # TODO Write down out how this works, once and for all...
    rho_0_vec = convert(MPS, rho_0)
    for i in 1:length(rho_0_vec)
        rho_0_vec[i] *= combiner(sites[i], sites[i]')
    end
    u_mat = 1 / sqrt(2) .* [
        1 0 0 1
        0 1 im 0
        0 1 -im 0
        -1 0 0 1
    ]
    combined_sites = siteinds(rho_0_vec)
    # The combiner produces an Index of type `(dim=4|id=...|"CMB,Link")`, but we
    # need a "vFermion" Index on each site instead. Let's also change that.
    u = [ITensor(ComplexF64, u_mat, combined_sites[i], vsites[i]) for i in 1:length(vsites)]
    for i in 1:length(rho_0_vec)
        rho_0_vec[i] *= u[i]
    end
    # Now we have a vectorized density matrix which we can use, with our
    # machinery, as an initial state.

    # Input: system parameters
    system_length = 1
    eps = parameters["sys_en"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    emptycoups = thermofield_coefficients[:, 1]
    emptyfreqs = thermofield_coefficients[:, 3]
    filledcoups = thermofield_coefficients[:, 2]
    filledfreqs = thermofield_coefficients[:, 4]

    chain_length = parameters["chain_length"]
    total_size = 2 * chain_length + 1
    systempos = 1
    filledchain_sites = 3:2:total_size
    emptychain_sites = 2:2:total_size
    total_size != length(psi_0) && error(
        "Number of sites between initial state from file and parameters does not match."
    )

    adjℓ = -eps * gkslcommutator("n", systempos)

    adjℓ +=
        emptycoups[1] *
        exchange_interaction′(vsites[systempos], vsites[emptychain_sites[1]]) +
        filledcoups[1] *
        exchange_interaction′(vsites[systempos], vsites[filledchain_sites[1]])
    +
    spin_chain′(
        emptyfreqs[1:chain_length], emptycoups[2:chain_length], vsites[emptychain_sites]
    ) + spin_chain′(
        filledfreqs[1:chain_length], filledcoups[2:chain_length], vsites[filledchain_sites]
    )

    adjL = MPO(adjℓ, vsites)

    opstrings = Dict(
        [
            systempos => "vAdag"
            [st => "vId" for st in filledchain_sites]
            [st => "vId" for st in emptychain_sites]
        ],
    )
    targetop = MPS(ComplexF64, vsites, opstrings)
    growMPS!(targetop, parameters["max_bond"])

    # The Green's function G(t) = -i ⟨ψ, d(t) d† ψ⟩ can be obtained by looking at the
    # "exp_val_im" column of the output file.

    adjtdvp1vec!(
        targetop,
        rho_0_vec,
        adjL,
        parameters["tstep"],
        parameters["tmax"],
        parameters["ms_stride"] * parameters["tstep"],
        vsites;
        progress=true,
        exp_tol=parameters["exp_tol"],
        krylovdim=parameters["krylov_dim"],
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
    )
end
