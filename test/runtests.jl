using UQPotentials
using Test
using Distributions

@testset "UQPotentials.jl" begin
    # Write your tests here.

    @test log_param_priors([0.0],[Normal(0.0,1.0)]) ≈ -0.9189 atol=0.0001

end
