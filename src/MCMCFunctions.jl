# module MCMCFunctions

# using Turing, Plots, StatsPlots

export determine_step_size_direct,determine_step_size_gp,rescale_chain,scale_parameters,generate_GP_Turing_distribution_morse,generate_GP_Turing_distribution,generate_GP_Turing_distribution_morse2,grw_MH_mcmc_gp

"""
    determine_step_size_direct(proposal_list,mcmc_func,x0,log_posterior,test_steps,parameter_names,min_val,max_val,basic_system_info,priors,target_dist_test,output_function)

Computes acceptance rates from different step sizes in proposal list.
"""
function determine_step_size_direct(proposal_list,mcmc_func,x0,log_posterior,test_steps,parameter_names,min_val,max_val,basic_system_info,priors,target_dist_test,output_function)
    for sigma_p in proposal_list 
        # chain, accept = mcmc_fs.grw_metropolis_mcmc_test(x0,log_posterior,499,sigma_p,parameter_names,
        #                                                 min_val,max_val,basic_system_info,priors_test,
        #                                                 target_dist_test)

        chain, accept = mcmc_func(x0,log_posterior,test_steps-1,sigma_p,parameter_names,
                                                        min_val,max_val,basic_system_info,priors,
                                                        target_dist_test,output_function)


        print("\n step size scale:",sigma_p)
        print("\n acceptance rate:",accept)
        print()
    end
    return
end

"""
    determine_step_size_gp(proposal_list)

Computes acceptance rates from different step sizes in proposal list. (check this WIP)
"""
function determine_step_size_gp(proposal_list,mcmc_func,x0, log_posterior_gp, test_steps,priors,target_dist,max_val,min_val,gp_surrogate)
    for sigma_p in proposal_list
        chain, accept = mcmc_func(x0, log_posterior_gp, test_steps-1, sigma_p,priors,
                                                            target_dist,max_val,min_val,gp_surrogate)
        print("\n step size scale:",sigma_p)
        print("\n acceptance rate:",accept)
        print()
    end
    return
end

"""
    rescale_chain(chain,max_val,min_val)

Rescales parameters in MCMC traces/chains.
"""
function rescale_chain(chain,max_val,min_val)
    chain_ = copy(chain) 
    for (count,i) in enumerate(chain_[:,1])
        for j in range(1,stop=size(chain)[2])
            chain_[count,j] = chain[count,j]*((max_val[j] - min_val[j])) + min_val[j]
        end
    end
    return chain_
end

##### Handwritten sampler functions (rewrite from python) #################################################################################################

"""
    grw_MH_mcmc_gp(x0, log_h, N, sigma,min_val,max_val,priors,target_dist,gaussian_process)

Gaussian random walk Metropolis-Hastings algorithm using gp surrogate. Rewritten from python (WIP). For single observation / data point
"""
function grw_MH_mcmc_gp(x0, log_h, N, sigma,min_val,max_val,priors,target_dist,gaussian_process)
    @assert ndims(x0) ==1
    d=length(x0)
    X = zeros((N,d))
    X[1,:] = x0
    accepted = 0
    log_hp = log_h(x0,priors,target_dist,max_val,min_val,gaussian_process) 
    
    for i in 2:N
        # generation
        Xn = X[i-1,:] + sigma*randn(d)
        
        # calculation
        log_hn = log_h(Xn,priors,target_dist,max_val,min_val,gaussian_process)
        alpha = min(1,exp(log_hn - log_hp))
        
        # accept / reject
        if rand(1)[1] <= alpha
            X[i,:] = Xn
            log_hp = log_hn
            accepted += 1
        else
            X[i,:] = X[i-1,:]
        end 
    end
    return X,accepted / N
end

"""
    grw_MH_mcmc_gp(x0, log_h, N, sigma,min_val,max_val,priors,data_means,data_st_devs,gaussian_process)

Gaussian random walk Metropolis-Hastings algorithm using gp surrogate. Rewritten from python (WIP). For vector of observations / data points
"""
function grw_MH_mcmc_gp(x0, log_h, N, sigma,min_val,max_val,priors,data_means,data_st_devs,gaussian_process)
    @assert ndims(x0) ==1
    d=length(x0)
    X = zeros((N,d))
    X[1,:] = x0
    accepted = 0
    log_hp = log_h(x0,priors,data_means,data_st_devs,max_val,min_val,gaussian_process) 
    
    for i in 2:N
        # generation
        Xn = X[i-1,:] + sigma*randn(d)
        
        # calculation
        log_hn = log_h(Xn,priors,data_means,data_st_devs,max_val,min_val,gaussian_process)
        alpha = min(1,exp(log_hn - log_hp))
        
        # accept / reject
        if rand(1)[1] <= alpha
            X[i,:] = Xn
            log_hp = log_hn
            accepted += 1
        else
            X[i,:] = X[i-1,:]
        end 
    end
    return X,accepted / N
end


##### Blackbox sampler functions ##########################################################################################################################

"""
    scale_parameters(X,max_val,min_val)

Scales parameters to similar magnitude for use in MCMC sampling. 
"""
function scale_parameters(X,max_val,min_val)
    X_s = zero(X)
    for j in range(1,stop=size(X)[2])
        X_s[:, j] = (X[:, j] .- min_val[j]) / (max_val[j] - min_val[j])
    end
    return X_s
end


"""
    generate_GP_Turing_distribution_morse(mcmc_step_num,original_parameter_values,output_function_info)

Generates GP / Turing distribution. Specifically for morse potential ONLY because we sampled from positive distribution, and required to flip sign of first parameter of potential for correct evaluations. (WIP)

### Arguments

- `mcmc_step_num`: number of steps required in MCMC chain
- `original_parameter_values`: list of nominal parameter values
- `output_function_info`: list of various objects defined by choice of output function in this order: basic function, turing model, reference kde of training data output (from LHC?), gp surrogate for specific output, and target distribution of output.

"""
function generate_GP_Turing_distribution_morse(mcmc_step_num,original_parameter_values,output_function_info)

    func,mcmc_model,reference_kde,gp_surrogate,target_dist = output_function_info

    chain = sample(mcmc_model,MH(),mcmc_step_num,init_theta=original_parameter_values)
    
    chain_params = chain.value.data[:,1:length(original_parameter_values)]
    #chain_params = chain.value.data[:,2:length(original_parameter_values)+1]
    chain_params[:,1] = -chain_params[:,1] # reverse sign for negative parameter

    out_list_main = fill(-1.0,mcmc_step_num)

    @time for i in 1:mcmc_step_num
        out_list_main[i] = mean(gp_surrogate([chain_params[i,:]]))[1]
    end
    
    p_main = Plots.plot(reference_kde,label="LHC priors",linestyle=:dash,lw=1.7,legend=:topleft)
    StatsPlots.plot!(p_main,target_dist,label="Likelihood target",linestyle=:dash)
    density!(p_main,out_list_main,label="MCMC, GP/Turing, $mcmc_step_num steps")
    xlabel!("QoI")
    ylabel!("Density")
    display(p_main)

    # kde_main = kde(out_list_main,npoints=n_points)

    # KL_div = gkl_divergence(reference_kde.x,kde_main.x);
    # @show KL_div

    return out_list_main,chain
    
end

function generate_GP_Turing_distribution_morse2(mcmc_step_num,original_parameter_values,output_function_info)

    func,mcmc_model,reference_kde,gp_surrogate,target_dist = output_function_info

    chain = sample(mcmc_model,MH(),mcmc_step_num,init_theta=original_parameter_values)
    
    #chain_params = chain.value.data[:,1:length(original_parameter_values)]
    chain_params = chain.value.data[:,2:length(original_parameter_values)+1]
    chain_params[:,1] = -chain_params[:,1] # reverse sign for negative parameter

    out_list_main = fill(-1.0,mcmc_step_num)

    @time for i in 1:mcmc_step_num
        out_list_main[i] = mean(gp_surrogate([chain_params[i,:]]))[1]
    end
    
    p_main = Plots.plot(reference_kde,label="LHC priors",linestyle=:dash,lw=1.7,legend=:topleft)
    StatsPlots.plot!(p_main,target_dist,label="Likelihood target",linestyle=:dash)
    density!(p_main,out_list_main,label="MCMC, GP/Turing, $mcmc_step_num steps")
    xlabel!("QoI")
    ylabel!("Density")
    display(p_main)

    # kde_main = kde(out_list_main,npoints=n_points)

    # KL_div = gkl_divergence(reference_kde.x,kde_main.x);
    # @show KL_div

    return out_list_main,chain
    
end

"""
    generate_GP_Turing_distribution_morse(mcmc_step_num,original_parameter_values,output_function_info)

Generates GP / Turing distribution.

### Arguments

- `mcmc_step_num`: number of steps required in MCMC chain
- `original_parameter_values`: list of nominal parameter values
- `output_function_info`: list of various objects defined by choice of output function in this order: basic function, turing model, reference kde of training data output (from LHC?), gp surrogate for specific output, and target distribution of output.

"""
function generate_GP_Turing_distribution(mcmc_step_num,original_parameter_values,output_function_info)

    func,mcmc_model,reference_kde,gp_surrogate,target_dist = output_function_info

    chain = sample(mcmc_model,MH(),mcmc_step_num,init_theta=original_parameter_values)
    
    chain_params = chain.value.data[:,1:length(original_parameter_values)]

    out_list_main = fill(-1.0,mcmc_step_num)

    @time for i in 1:mcmc_step_num
        out_list_main[i] = mean(gp_surrogate([chain_params[i,:]]))[1]
    end
    
    p_main = Plots.plot(reference_kde,label="LHC priors",linestyle=:dash,lw=1.7,legend=:topleft)
    StatsPlots.plot!(p_main,target_dist,label="Likelihood target",linestyle=:dash)
    density!(p_main,out_list_main,label="MCMC, GP/Turing, $mcmc_step_num steps")
    xlabel!("QoI")
    ylabel!("Density")
    display(p_main)

    # kde_main = kde(out_list_main,npoints=n_points)

    # KL_div = gkl_divergence(reference_kde.x,kde_main.x);
    # @show KL_div

    return out_list_main,chain
    
end

# end