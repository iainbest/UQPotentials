# module ExtraPlotFunctions

# using Plots,StatsPlots

export trace_plot,compare_distributions,output_comparison_plot

"""
    trace_plot(chain)

Plot traces for given mcmc chain. 

Plots parameter traces on same graph - WIP to modify to subplots?
"""
function trace_plot(chain)
    display(Plots.plot(chain))
end

"""
    compare_distributions(priors,chain_list)

Take parameter priors and list of MCMC chains, and plot prior and posterior distributions for parameters
(i.e. compare input parameter distributions).

### Arguments

- `priors`: Prior distributions of parameters as list
- `chain_list`: List of MCMC chains for separate parameters

"""
function compare_distributions(priors,chain_list)
    for (count,prior) in enumerate(priors)
        plot = Plots.plot(prior,label="prior",normalize=true)
        for chain in chain_list
            density!(plot,chain[:,count],normalize=true)  
        end
        display(plot) 
    end  
end

# to plot output (e.g. B) distributions
"""
    output_comparison_plot(reference_kde,target_dist,surrogate_data,mcmc_step_num,out_lists,labels = ["Direct / Handwritten","Direct / Turing","GP / Handwritten","GP / Turing"])

Compare distributions of output quantity for different MCMC methods. (WIP)

### Arguments

- `reference_kde`: kde of reference LHC output quantity
- `target_dist`: target distribution of output quantity
- `surrogate_data`: kde of output of data the GP surrogate was trained on
- `mcmc_step_num`: number of MCMC steps performed for all methods
- `out_lists`: list of output lists from different MCMC methods
- `labels = ["Direct / Handwritten","Direct / Turing","GP / Handwritten","GP / Turing"]`: list describing different MCMC methods

"""
function output_comparison_plot(reference_kde,target_dist,surrogate_data,mcmc_step_num,out_lists,labels = ["Direct / Handwritten","Direct / Turing","GP / Handwritten","GP / Turing"])


    p1 = Plots.plot(reference_kde,label="LHC data",linestyle=:dash,lw=1.7)
    StatsPlots.plot!(target_dist,label="Target Likelihood",linestyle=:dash) # target / likelihood dist
    density!(p1,surrogate_data,label="GP Surrogate data",normalize=true,linestyle=:dash,lw=1.7) #surrogate dist
    xlabel!("Output quantity")
    ylabel!("Density")
    title!("Output distribution comparison, $mcmc_step_num samples")
    
    for (count,i) in enumerate(out_lists)
        density!(p1,i,label=labels[count],linealpha=0.75) 
    end
    display(p1)
end

# end