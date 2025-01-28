using OpenSystemsChainMapping
using Test

include("qns.jl")

@testset "Quantum numbers are set correctly" begin
    @test siam_spinless_pure_state_hasqns()
    @test siam_spinless_superfermions_mc_hasqns()
end

@testset "MPS link enlargement with QNs" begin
    @test siam_spinless_pure_state_qns_enlarge()
    @test siam_spinless_superfermions_mc_qns_enlarge()
end
