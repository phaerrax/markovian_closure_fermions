export tdvp!, tdvpMC!, tdvp1!

using ITensors: position!

zerosite!(PH::ProjMPO) = (PH.nsite = 0)
singlesite!(PH::ProjMPO) = (PH.nsite = 1)
twosite!(PH::ProjMPO) = (PH.nsite = 2)

struct TDVP1 end
struct TDVP2 end

"""
    tdvp!(psi,H::MPO,dt,tf; kwargs...)
Evolve the MPS `psi` up to time `tf` using the two-site time-dependent variational
principle as described in Ref. [1].

# Keyword arguments:
All keyword arguments controlling truncation which are accepted by ITensors.replaceBond!,
namely:
- `maxdim::Int`: If specified, keep only `maxdim` largest singular values after applying gate.
- `mindim::Int`: Minimal number of singular values to keep if truncation is performed according to
    value specified by `cutoff`.
- `cutoff::Float`: If specified, keep the minimal number of singular-values such that the discarded weight is
    smaller than `cutoff` (but bond dimension will be kept smaller than `maxdim`).
- `absoluteCutoff::Bool`: If `true` truncate all singular-values whose square is smaller than `cutoff`.

In addition the following keyword arguments are supported:
- `hermitian::Bool` (`true`) : whether the MPO `H` represents an Hermitian operator. This will be passed to the
    Krylov exponentiation routine (`KrylovKit.exponentiate`) which will in turn use a Lancosz algorithm in the
    case of an hermitian operator.
- `exp_tol::Float` (1e-14) : The error tolerance for `KrylovKit.exponentiate`.
    (note that default value was not optimized yet, so you might want to play around with it)
- `progress::Bool` (`true`) : If `true` a progress bar will be displayed

# References:
[1] Haegeman, J., Lubich, C., Oseledets, I., Vandereycken, B., & Verstraete, F. (2016).
Unifying time evolution and optimization with matrix product states. Physical Review B, 94(16).
https://doi.org/10.1103/PhysRevB.94.165116
"""
function tdvp!(psi,H::MPO,dt,tf; kwargs...)
    nsteps = Int(tf/dt)
    cb = get(kwargs,:callback, NoTEvoCallback())
    hermitian = get(kwargs,:hermitian,true)
    exp_tol = get(kwargs,:exp_tol, 1e-14)
    krylovdim = get(kwargs,:krylovdim, 30 )
    maxiter = get(kwargs,:maxiter,100)
    normalize = get(kwargs,:normalize,true)

    io_file = get(kwargs,:io_file,nothing)
    ranks_file = get(kwargs,:io_ranks,nothing)
    times_file = get(kwargs,:io_times,nothing)
    
    pbar = get(kwargs,:progress, true) ? Progress(nsteps, desc="Evolving state... ") : nothing
    τ = 1im*dt
    store_psi0 = get(kwargs,:store_psi0,false)
    imag(τ) == 0 && (τ = real(τ))

    #Smart sintax...
    store_psi0 && (psi0 = copy(psi));

    #If present, open measurements file.
    #The store_psi0 triggers the measurement of the overlap
    if(io_file != nothing)
        io_handle = open(io_file,"w");

        #Write column names to file
        @printf(io_handle,"#%19s", "time")
        res = measurements(cb)
        for o in sort(collect(keys(res)))
            @printf(io_handle,"%40s",o)
        end
        if(store_psi0)
            @printf(io_handle,"%40s%40s","re_over","im_over")
        end
        @printf(io_handle,"%40s", "Norm")
        @printf(io_handle,"\n")

    end

    if(ranks_file != nothing)
        ranks_handle = open(ranks_file,"w");

        #Write column names to file
        @printf(ranks_handle,"#%19s", "time")
        for o in 1:length(psi)-1
            @printf(ranks_handle,"%10d",o)
        end
        
        @printf(ranks_handle,"\n")

    end

    if(times_file != nothing)
        times_handle = open(times_file,"w");

        #Write column names to file
        @printf(times_handle,"#%19s", "walltime (sec)")
        @printf(times_handle,"\n")

    end

    N = length(psi)
    orthogonalize!(psi,1)
    PH = ProjMPO(H)
    position!(PH,psi,1)
    
    for s in 1:nsteps
        stime = @elapsed begin
        for (b,ha) in sweepnext(N)
            #evolve with two-site Hamiltonian
            twosite!(PH)
            ITensors.position!(PH,psi,b)
            wf = psi[b]*psi[b+1]
            wf, info = exponentiate(PH, -τ/2, wf; ishermitian=hermitian , tol=exp_tol, krylovdim=krylovdim)
            dir = ha==1 ? "left" : "right"
            info.converged==0 && throw("exponentiate did not converge")
            spec = replacebond!(psi,b,wf;normalize=normalize, ortho = dir, kwargs... )
            # normalize && ( psi[dir=="left" ? b+1 : b] /= sqrt(sum(eigs(spec))) )

            apply!(cb,psi; t=s*dt,
                   bond=b,
                   sweepend= ha==2,
                   sweepdir= ha==1 ? "right" : "left",
                   spec=spec,
                   alg=TDVP2())

            # evolve with single-site Hamiltonian backward in time.
            # In the case of imaginary time-evolution this step
            # is not necessary (see Ref. [1])
            i = ha==1 ? b+1 : b
            if 1<i<N && !(dt isa Complex)
                singlesite!(PH)
                ITensors.position!(PH,psi,i)
                psi[i], info = exponentiate(PH,τ/2,psi[i]; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim,
                                            maxiter=maxiter)
                info.converged==0 && throw("exponentiate did not converge")
            elseif i==1 && dt isa Complex
                # TODO not sure if this is necessary anymore
                psi[i] /= sqrt(real(scalar(dag(psi[i])*psi[i])))
            end

        end
        
        #this end ends @elapsed
        end
        
        !isnothing(pbar) && ProgressMeter.next!(pbar, showvalues=[("t", dt*s),("dt step time", round(stime,digits=3)),("Max bond-dim", maxlinkdim(psi))]);

        #if there is output file and the time is right...
        if(!isnothing(io_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
                #Appo variable
                res = measurements(cb);
                @printf(io_handle,"%40.15f",measurement_ts(cb)[end])
                for o in sort(collect(keys(res)))
                    @printf(io_handle,"%40.15f",res[o][end][1])
                end

                if(store_psi0)
                    over = dot(psi0,psi);

                    @printf(io_handle,"%40s.15f%40.15f",real(over),imag(over))
                end

                #Print Norm
                @printf(io_handle,"40.15f",norm(psi))
                @printf(io_handle,"\n")
                flush(io_handle)
            end
        end

        #Ranks printout
        if(!isnothing(ranks_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
            
                @printf(ranks_handle,"%40.15f",measurement_ts(cb)[end])
            
                pluto = [dim(linkind(psi,j)) for j in 1:length(psi)-1]
                for o in pluto
                    @printf(ranks_handle,"%10d",o)
                end

                @printf(ranks_handle,"\n")
                flush(ranks_handle)
            end
        end

        if(!isnothing(times_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
            
                @printf(times_handle,"%20.4f\n",stime)
                flush(times_handle)
            end
        end

        checkdone!(cb) && break 
    end
    if(!isnothing(io_file))
        close(io_handle)
    end

    if(!isnothing(ranks_file))
        close(ranks_handle)
    end

    if(!isnothing(times_file))
        close(times_handle)
    end

end
function tdvpMC!(psi,H::MPO,dt,tf; kwargs...)
    nsteps = Int(tf/dt)
    cb = get(kwargs,:callback, NoTEvoCallback())
    #Default value to false (not Hermitian)
    hermitian = get(kwargs,:hermitian,false)
    exp_tol = get(kwargs,:exp_tol, 1e-14)
    krylovdim = get(kwargs,:krylovdim, 30 )
    maxiter = get(kwargs,:maxiter,100)
    #Default value to false (don't normalize)
    normalize = get(kwargs,:normalize,false)

    io_file = get(kwargs,:io_file,nothing)
    ranks_file = get(kwargs,:io_ranks,nothing)
    times_file = get(kwargs,:io_times,nothing)
    
    pbar = get(kwargs,:progress, true) ? Progress(nsteps, desc="Evolving state... ") : nothing

#Preamble
#println(BLAS.get_config());
#print("NumThreads: ");

#println(BLAS.get_num_threads())
#flush(stdout)




    τ = 1im*dt
    store_psi0 = get(kwargs,:store_psi0,false)
    imag(τ) == 0 && (τ = real(τ))

    #Smart sintax...
    store_psi0 && (psi0 = copy(psi));

    #If present, open measurements file.
    #The store_psi0 triggers the measurement of the overlap
    if(io_file != nothing)
        io_handle = open(io_file,"w");

        #Write column names to file
        @printf(io_handle,"#%19s", "time")
        res = measurements(cb)
        for o in sort(collect(keys(res)))
            @printf(io_handle,"%40s",o)
        end


        if(store_psi0)
            @printf(io_handle,"%40s%40s","re_over","im_over")
        end

        @printf(io_handle,"%40s","Norm")

        @printf(io_handle,"\n")

    end

    if(ranks_file != nothing)
        ranks_handle = open(ranks_file,"w");

        #Write column names to file
        @printf(ranks_handle,"#%19s", "time")
        for o in 1:length(psi)-1
            @printf(ranks_handle,"%10d",o)
        end
        
        @printf(ranks_handle,"\n")

    end

    if(times_file != nothing)
        times_handle = open(times_file,"w");

        #Write column names to file
        @printf(times_handle,"#%19s", "walltime (sec)")
        @printf(times_handle,"\n")

    end

    N = length(psi)
    orthogonalize!(psi,1)
    PH = ProjMPO(H)
    position!(PH,psi,1)
    
    for s in 1:nsteps
        stime = @elapsed begin
        for (b,ha) in sweepnext(N)
            #evolve with two-site Hamiltonian
            twosite!(PH)
            ITensors.position!(PH,psi,b)
            wf = psi[b]*psi[b+1]
            wf, info = exponentiate(PH, -τ/2, wf; ishermitian=hermitian , tol=exp_tol, krylovdim=krylovdim)
            dir = ha==1 ? "left" : "right"
            info.converged==0 && throw("exponentiate did not converge")
            spec = replacebond!(psi,b,wf;normalize=normalize, ortho = dir, kwargs... )
            # normalize && ( psi[dir=="left" ? b+1 : b] /= sqrt(sum(eigs(spec))) )

            apply!(cb,psi; t=s*dt,
                   bond=b,
                   sweepend= ha==2,
                   sweepdir= ha==1 ? "right" : "left",
                   spec=spec,
                   alg=TDVP2())

            # evolve with single-site Hamiltonian backward in time.
            # In the case of imaginary time-evolution this step
            # is not necessary (see Ref. [1])
            i = ha==1 ? b+1 : b
            if 1<i<N && !(dt isa Complex)
                singlesite!(PH)
                ITensors.position!(PH,psi,i)
                psi[i], info = exponentiate(PH,τ/2,psi[i]; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim,
                                            maxiter=maxiter)
                info.converged==0 && throw("exponentiate did not converge")
            elseif i==1 && dt isa Complex
                # TODO not sure if this is necessary anymore
                psi[i] /= sqrt(real(scalar(dag(psi[i])*psi[i])))
            end

        end
        
        #this end ends @elapsed
        end
        
        !isnothing(pbar) && ProgressMeter.next!(pbar, showvalues=[("t", dt*s),("dt step time", round(stime,digits=3)),("Max bond-dim", maxlinkdim(psi))]);

        #if there is output file and the time is right...
        if(!isnothing(io_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
                #Appo variable
                res = measurements(cb);
                @printf(io_handle,"%40.15f",measurement_ts(cb)[end])
                for o in sort(collect(keys(res)))
                    @printf(io_handle,"%40.15f",res[o][end][1])
                end

                if(store_psi0)
                    over = dot(psi0,psi);

                    @printf(io_handle,"%40.15f%40.15f",real(over),imag(over))
                end
                
                #Print norm
                @printf(io_handle,"%40.15f",norm(psi))
                @printf(io_handle,"\n")
                flush(io_handle)
            end
        end

        #Ranks printout
        if(!isnothing(ranks_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
            
                @printf(ranks_handle,"%40.15f",measurement_ts(cb)[end])
            
                pluto = [dim(linkind(psi,j)) for j in 1:length(psi)-1]
                for o in pluto
                    @printf(ranks_handle,"%10d",o)
                end

                @printf(ranks_handle,"\n")
                flush(ranks_handle)
            end
        end

        if(!isnothing(times_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
            
                @printf(times_handle,"%20.4f\n",stime)
                flush(times_handle)
            end
        end

        checkdone!(cb) && break 
    end
    if(!isnothing(io_file))
        close(io_handle)
    end

    if(!isnothing(ranks_file))
        close(ranks_handle)
    end

    if(!isnothing(times_file))
        close(times_handle)
    end

end

function tdvp1!(psi,H::MPO,dt,tf; kwargs...)
    
    nsteps = Int(tf/dt)
    cb = get(kwargs,:callback, NoTEvoCallback())
    hermitian = get(kwargs,:hermitian,true)
    exp_tol = get(kwargs,:exp_tol, 1e-14)
    krylovdim = get(kwargs,:krylovdim, 30 )
    maxiter = get(kwargs,:maxiter,100)
    normalize = get(kwargs,:normalize,true)

    io_file = get(kwargs,:io_file,nothing)
    ranks_file = get(kwargs,:io_ranks,nothing)
    times_file = get(kwargs,:io_times,nothing)
    
    pbar = get(kwargs,:progress, true) ? Progress(nsteps, desc="Evolving state... ") : nothing
    τ = dt
    store_psi0 = get(kwargs,:store_psi0,false)
    imag(τ) == 0 && (τ = real(τ))

    #Smart sintax...
    store_psi0 && (psi0 = copy(psi));

    #If present, open measurements file.
    #The store_psi0 triggers the measurement of the overlap
    if(io_file != nothing)
        io_handle = open(io_file,"w");

        #Write column names to file
        @printf(io_handle,"#%19s", "time")
        res = measurements(cb)
        for o in sort(collect(keys(res)))
            @printf(io_handle,"%40s",o)
        end
        if(store_psi0)
            @printf(io_handle,"%40s%40s","re_over","im_over")
        end
        @printf(io_handle,"%40s", "Norm")
        @printf(io_handle,"\n")

    end

    if(ranks_file != nothing)
        ranks_handle = open(ranks_file,"w");

        #Write column names to file
        @printf(ranks_handle,"#%19s", "time")
        for o in 1:length(psi)-1
            @printf(ranks_handle,"%10d",o)
        end
        
        @printf(ranks_handle,"\n")

    end

    if(times_file != nothing)
        times_handle = open(times_file,"w");

        #Write column names to file
        @printf(times_handle,"#%19s", "walltime (sec)")
        @printf(times_handle,"\n")

    end

    N = length(psi)

    #Prepare for first iteration
    orthogonalize!(psi,1)
    PH = ProjMPO(H)
    singlesite!(PH)
    position!(PH,psi,1)
    
    for s in 1:nsteps
        stime = @elapsed begin
            #Pay attention to iterator:
            #ncenter determines the end and turning points of the loop
            
            #In TDVP1 b means site, not bond!
            for (b,ha) in sweepnext(N,ncenter = 1)
                #evolve with one-site Hamiltonian
                singlesite!(PH)

                #postime = @elapsed begin
                ITensors.position!(PH,psi,b)
                #end
                #println("tempo position b",postime)
                #Here we could merge TDVP1 and TDVP2
                phi1 = psi[b]
                #Evolve local tensor

                #Debug
                #println("Forward time evolution: b, ha ",b, ha);

                #exptime = @elapsed begin
                phi1, info = exponentiate(PH, -τ/2, phi1; ishermitian=hermitian , tol=exp_tol, krylovdim=krylovdim, maxiter = maxiter, eager = true)
                #end
                #println("Tempo esponenziazione avanti: ", exptime)

                #println("Froward done")
                #Replace (temporarily) the local tensor with the evolved one.
                
                info.converged==0 && throw("exponentiate did not converge")

                psi[b] = phi1
                #in this moment the 
                #mtime = @elapsed begin
                
                #Check measurement faulty    
                apply!(cb,psi; t=s*dt,
                bond=b,
                sweepend= ha==2,
                sweepdir= ha==1 ? "right" : "left",
                #spec=spec,
                alg=TDVP1())
                #end mtime
                #end
                #println("Tempo per misure: ", mtime)

                #Copypasted from
                #https://github.com/ITensor/ITensorTDVP.jl/blob/main/src/tdvp_step.jl
                
                if (ha == 1 && (b != N)) || (ha == 2 && b != 1)
                    b1 = (ha == 1 ? b + 1 : b)
                    Δ = (ha == 1 ? +1 : -1)
                    #left bond intex and physical index
                    uinds = uniqueinds(phi1, psi[b + Δ])

                 #   svdtime = @elapsed begin
                    U, S, V = svd(phi1, uinds)
                  #  end
                  #  println("Tempo svd: ", svdtime)

                    #the tensor in b is now left/right orthogonal
                    psi[b] = U
                    phi0 = S * V
                    if ha == 1
                        ITensors.setleftlim!(psi, b)
                    elseif ha == 2
                        ITensors.setrightlim!(psi, b)
                    end
                    
                    zerosite!(PH)
                    #postime = @elapsed begin
                    position!(PH, psi, b1)
                    #end
                    #println("tempo position b+1", postime)

                    #Debug
                    #println("Backward-time b, ha", b, ha)
                    #exptime = @elapsed begin
                    phi0, info = exponentiate(PH, τ/2, phi0; ishermitian=hermitian , tol=exp_tol, krylovdim=krylovdim, maxiter = maxiter,eager = true)
                    #end
                    #println("Tempo esponenziazione indietro: ", exptime)
                    #println("backward done!")
                   
                    psi[b + Δ] = phi0 * psi[b + Δ]
                    
                    if ha == 1
                        ITensors.setrightlim!(psi, b + Δ + 1)
                    elseif ha == 2
                        ITensors.setleftlim!(psi, b + Δ - 1)
                    end                    
                    
                    singlesite!(PH)
                end

                #Adesso il passo di evoluzione e` finito.
                #Considerando che misuriamo "durante" lo sweep 
                #a sx, adesso l'ortogonality center e` sul sito a sx del bond
                #quindi credo che la misura vada spostata piu` su
                #ovvero prima di rendere il tensore right orthogonal

               
    
            end
            
            #this end ends @elapsed
        end
        
        #println("time-step time: ",s*dt," t=", stime)
        !isnothing(pbar) && ProgressMeter.next!(pbar, showvalues=[("t", dt*s),("dt step time", round(stime,digits=3)),("Max bond-dim", maxlinkdim(psi))]);

        #if there is output file and the time is right...
        if(!isnothing(io_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
                #Appo variable
                res = measurements(cb);
                @printf(io_handle,"%40.15f",measurement_ts(cb)[end])
                for o in sort(collect(keys(res)))
                    @printf(io_handle,"%40.15f",res[o][end][1])
                end

                if(store_psi0)
                    over = dot(psi0,psi);

                    @printf(io_handle,"%40.15f%40.15f",real(over),imag(over))
                end

                #Print Norm
                # print("Norm: ")
                # println(norm(psi))
                @printf(io_handle,"%40.15f", norm(psi))
                @printf(io_handle,"\n")
                flush(io_handle)
            end
        end

        #Ranks printout
        if(!isnothing(ranks_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
            
                @printf(ranks_handle,"%40.15f",measurement_ts(cb)[end])
            
                pluto = [dim(linkind(psi,j)) for j in 1:length(psi)-1]
                for o in pluto
                    @printf(ranks_handle,"%10d",o)
                end

                @printf(ranks_handle,"\n")
                flush(ranks_handle)
            end
        end

        if(!isnothing(times_file) && length(measurement_ts(cb))>0 )
        
            if ( dt*s ≈ measurement_ts(cb)[end])
            
                @printf(times_handle,"%20.4f\n",stime)
                flush(times_handle)
            end
        end

        checkdone!(cb) && break 
    end
    if(!isnothing(io_file))
        close(io_handle)
    end

    if(!isnothing(ranks_file))
        close(ranks_handle)
    end

    if(!isnothing(times_file))
        close(times_handle)
    end

end