# module UtilityFunctions

# using Statistics

export characterise_uncertainty

function characterise_uncertainty(QoI_list,ref_value=nothing,ref_stdev=nothing)
    QoI_mean = mean(QoI_list)
    QoI_stdev = std(QoI_list)
    
    if ref_value !== nothing && ref_stdev !== nothing
        print("\n Reference result = ", ref_value," ± ",2.0*ref_stdev)
        print("\n Output result = ", QoI_mean," ± ",2.0*QoI_stdev)
    elseif ref_value !== nothing
        print("\n Reference result = ", ref_value)
        print("\n Output result = ", QoI_mean," ± ",2.0*QoI_stdev)
    else
        print("\n Output result = ", QoI_mean," ± ",2.0*QoI_stdev)
    end
    
end

# end