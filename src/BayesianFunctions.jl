# module BayesianFunctions

# using AbstractGPs, Distributions

export log_param_priors,log_likelihood,log_posterior,log_posterior_gp

"""
    log_param_priors(x,priors)

Computes sum of (log) probability density functions of parameter priors at point x. 

### Arguments

- `x`: values of the parameters of a potential. List format
- `priors`: prior distributions of parameters of the potential. List of Distributions.jl univariate distributions (check!)

"""
function log_param_priors(x,priors)    # priors (distributions) UNSCALED / ACTUAL NUMBERS
    
    # test this! should flip epsilon value to positive so doesn't break distribution
    x = x
    x[1] = abs(x[1])
    
    log_prior_list = [logpdf(prior,i) for (i,prior) in zip(x,priors)]
    return sum(log_prior_list)
end

"""
    log_likelihood(output_value,target_dist)

Computes log likelihood of an output given a target distribution.

REMOVE THIS & USE loglikelihood FROM Distributions.jl  !! (may need to overload, so potentially keep this?)
"""
function log_likelihood(output_value::Float64,target_dist::UnivariateDistribution)   # target_dist UNSCALED
    return loglikelihood(target_dist,output_value)
end

"""
    log_likelihood(output_value,data_means,data_st_devs)

Constructs Normal distributions from input data means and standard deviations, then calculates log likelihood of one observation compared to (specific here??) those observations

### Arguments

- `output_value`: output value of simulation
- `data_means`: vector of mean values of observations / data
- `data_st_devs`: vector of standard deviations of observations / data


"""
function log_likelihood(output_value::Float64,data_means::Vector{Float64},data_st_devs::Vector{Float64})

    target_dists = fill(Normal(),length(data_means))
    
    for (count,(i,j)) in enumerate(zip(data_means,data_st_devs))
        target_dists[count] = Normal(i,j)
    end

    return sum(loglikelihood.(target_dists,output_value))
end

# takes SCALED / NOT ACTUAL NUMBERS input
# function log_posterior(x,priors,target_dist,max_val,min_val,parameter_names,basic_system_info)
#     x_s = (x .* (max_val .- min_val)) .+ min_val
#     bulk_eval = kim_fs.bulk_modulus(x_s,parameter_names,basic_system_info)
#     return log_param_priors(x_s,priors) + log_likelihood(bulk_eval,target_dist)
# end

"""
    log_posterior(x,priors,target_dist,max_val,min_val,parameter_names,basic_system_info,output_function)

Computes sum of log prior(s) and log likelihood, by first evaluating the output quantity via direct call to a function.

### Arguments

- `x`: scaled list of inputs corresponding to potential parameter values
- `priors`: prior distributions of parameters of the potential. List of Distributions.jl univariate distributions (check!)
- `target_dist`: target distribution of output quantity
- `max_val`: list of values greater than nominal, used to scale input to handwritten mcmc sampler. Chosen such that nominal values are equal to 0.5
- `min_val`: list of values greater than nominal, used to scale input to handwritten mcmc sampler.
- `parameter names`: List of parameter names 
- `basic_system_info`: List of basic system info in this order: species (e.g. "Ar" for Argon), crystal structure (e.g. "fcc", following ASE conventions), OpenKIM modelname string, and lattice parameter of system.
- `output_function`: Function which calculates quantity of interest

"""
function log_posterior(x,priors,target_dist,max_val,min_val,parameter_names,basic_system_info,output_function)
    x_s = (x .* (max_val .- min_val)) .+ min_val

    bulk_eval = output_function(x_s,parameter_names,basic_system_info)

    return log_param_priors(x_s,priors) + log_likelihood(bulk_eval,target_dist)
end

"""
    log_posterior_gp(x,priors,target_dist,max_val,min_val,gaussian_process)

Computes sum of log prior(s) and log likelihood, by first evaluating the output quantity via a Gaussian process surrogate.

### Arguments

- `x`: scaled list of inputs corresponding to potential parameter values
- `priors`: prior distributions of parameters of the potential. List of Distributions.jl univariate distributions (check!)
- `target_dist`: target distribution of output quantity
- `max_val`: list of values greater than nominal, used to scale input to handwritten mcmc sampler. Chosen such that nominal values are equal to 0.5
- `min_val`: list of values greater than nominal, used to scale input to handwritten mcmc sampler.
- `gaussian_process`: trained GP to calculate output quantity (assuming AbstractGPs GP)

"""
function log_posterior_gp(x::Vector{Float64},priors::Vector,target_dist::UnivariateDistribution,max_val::Vector{Float64},min_val::Vector{Float64},gaussian_process::AbstractGPs.PosteriorGP)
    x_s = (x .* (max_val .- min_val)) .+ min_val
    
    bulk_eval = mean(gaussian_process([x_s]))[1]
    
    return log_param_priors(x_s,priors) + log_likelihood(bulk_eval,target_dist)
end

"""
    log_posterior_gp(x,priors,data_means,data_st_devs,max_val,min_val,gaussian_process)
    
For vector of observations in log likelihood (WIP)
"""
function log_posterior_gp(x::Vector{Float64},priors::Vector,data_means::Vector{Float64},data_st_devs::Vector{Float64},max_val::Vector{Float64},min_val::Vector{Float64},gaussian_process::AbstractGPs.PosteriorGP)
    x_s = (x .* (max_val .- min_val)) .+ min_val
    
    bulk_eval = mean(gaussian_process([x_s]))[1]
    
    return log_param_priors(x_s,priors) + log_likelihood(bulk_eval,data_means,data_st_devs)
end

# end