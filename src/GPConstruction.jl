# module GPConstruction

# using KernelFunctions,AbstractGPs,Optim,Plots

export negativelogmarginallikelihood,loss_function,SEKernel_hyper_optim,build_GP_surrogate,check_GP_predictions

"""
    loss_function(x,y)

Loss function for negative log marginal likelihood (WIP)
"""
function loss_function(x, y)
    function negativelogmarginallikelihood(params)
        
        proposed_var = abs(params[1])
        proposed_lengthscales = abs.(params[2:end-1])
        
        kernel = proposed_var*(SEKernel()∘ARDTransform(1.0 ./ proposed_lengthscales))
                
        f = GP(kernel)
        fx = f(RowVecs(x), abs(params[end]))
        #fx = f(RowVecs(x), 1.0)
        return -logpdf(fx, y)
    end
    return negativelogmarginallikelihood
end

"""
    SEKernel_hyper_optim(X,Y,current_hypers,num_restarts,optim_iter)

Optimise hyperparameters of squared exponential kernel by minimising loss function (negative log marginal likelihood). (WIP)
"""
function SEKernel_hyper_optim(X,Y,current_hypers,num_restarts,optim_iter)

    # current_hypers contains kernel variance, then appropriate number of lengthscales, then noise/sigma

    # # show un-optimised hyperparam solution for reference/start
    kernel = current_hypers[1] * (SEKernel()∘ARDTransform(1.0 ./ current_hypers[2:end-1]))

    f = GP(kernel)
    fx = f(RowVecs(X), current_hypers[end])
    current_hypers_target = -logpdf(fx, Y)

    # verbose option?
    # @show current_hypers_target
    # print("\n")



    plot = Plots.plot()

    for i in 1:num_restarts

        θ0 = randn(length(current_hypers))

        # print trace in optim.options?

        opt = Optim.optimize(loss_function(X, Y), θ0, Optim.Options(iterations=optim_iter, store_trace=true))
        trace = Optim.trace(opt)

        objective = []
        for i in 1:length(trace)
            append!(objective, parse(Float64, split(string(trace[i]))[2]))
        end

        Plots.plot!(plot,objective,xlabel="iteration",ylabel="-logpdf",legend=false)
        

        #opt.minimizer[i] contains the i hyperparameters, in same format as current_hypers above

        kernel_variance = abs(opt.minimizer[1])
        kernel_lengthscales = abs.(opt.minimizer[2:end-1])
        sigma = abs(opt.minimizer[end])

        kernel = kernel_variance * (SEKernel()∘ARDTransform(1.0 ./ kernel_lengthscales))

        f = GP(kernel)
        fx = f(RowVecs(X), sigma)

        # prints for testing (add verbose option?)

    #     @show [kernel_variance,kernel_lengthscales,sigma]
    #     @show -logpdf(fx, Y)
    #     print("\n")

        if -logpdf(fx,Y) < current_hypers_target
            current_hypers_target = -logpdf(fx,Y)
            current_hypers = abs.(opt.minimizer)
        end

    end

    display(plot)

    return current_hypers
    
end

"""
    build_GP_surrogate(X,Y,hyperparam_list)

Build GP surrogate based on input parameters X (a matrix), output data at those parameters Y (a vector of observations), 
and a list of optimised hyperparameters in format [kernel variance,lengthscales(s),noise], where the kernel used is the 
squared exponential kernel. 
"""
function build_GP_surrogate(X,Y,hyperparam_list)

    kernel = hyperparam_list[1] * (SEKernel()∘ARDTransform(1.0 ./ hyperparam_list[2:end-1]))

    f = GP(kernel)
    fx = f(RowVecs(X), hyperparam_list[end])
    gp_surrogate = posterior(fx,Y);

    return gp_surrogate

end

"""
    check_GP_predictions(ell,output_function_id,param_index,original_parameter_values,parameter_names,basic_system_info,X,gp_surrogate,func)

Plots 1D slices of GP predictions, with data and confidence intervals. (WIP)

Fixes all but one of the potential parameters (defined by param_index). Compares GP predictions and direct calls to functions.
"""
function check_GP_predictions(ell,output_function_id,param_index,original_parameter_values,parameter_names,basic_system_info,X,gp_surrogate,func)

    # if output_function_id == "bulk_modulus"
    #     gp_surrogate = gp_surrogate_B
    #     func = kim_fs.bulk_modulus
    # elseif output_function_id == "vacancy_formation_energy"
    #     gp_surrogate = gp_surrogate_e_vac
    #     func = kim_fs.vacancy_formation_energy
    # else
    #     print("incorrect output id, retry")
    # end

    # range larger than trained range
    min_val__=0.85*original_parameter_values[param_index]
    max_val__=1.15*original_parameter_values[param_index]

    last_param_range = range(min_val__,max_val__,length=ell)

    out_test = fill(-1.0,ell)

    means = fill(-1.0,ell)
    vars = fill(-1.0,ell)

    for (count,i) in enumerate(last_param_range)
        #params = append!(original_parameter_values[1:end-1],i)
        params = copy(original_parameter_values)
        params[param_index] = i

        out_test[count] = func(params,parameter_names,basic_system_info)

        meanandvar = mean_and_var(gp_surrogate([params]))
        means[count] = meanandvar[1][1]
        vars[count] = meanandvar[2][1]
    end

    vars_ = 2.0.*sqrt.(vars)

    p1 = Plots.plot(last_param_range,means,ribbon=vars_,label="Mean GP prediction",xlabel=parameter_names[param_index],
                    ylabel="QoI")
    Plots.scatter!(p1,last_param_range[1:5:end],out_test[1:5:end],label="Direct evals")
    # range of values GP was trained on
    vline!(p1,[minimum(X[:,param_index]),maximum(X[:,param_index])],label="Training data range",ls=:dash)
    vline!(p1,[original_parameter_values[param_index]],label="nominal value",ls=:dash)
    # title!(p1,"GP vs Direct evaluations for changing parameter")
    display(p1)

    p2 = Plots.scatter(out_test,means,xlabel="True values",ylabel="GP prediction",legend=false)
    Plots.plot!(p2,[minimum(means),maximum(means)],[minimum(means),maximum(means)],ls=:dash)
    display(p2)

    p3 = Plots.scatter(abs.(means .- out_test),sqrt.(vars),xlabel="True error",ylabel="GP predicted error",
                        aspect_ratio=:equal,legend=false)
    Plots.plot!(p3,[minimum(sqrt.(vars)),maximum(sqrt.(vars))],[minimum(sqrt.(vars)),maximum(sqrt.(vars))],ls=:dash)
    # Plots.ylims!(p3,(0,0.002))
    # Plots.xlims!(p3,(0,0.002))
    display(p3)
    
end

# end