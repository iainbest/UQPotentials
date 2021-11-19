# module UtilityFunctions

# using Statistics

export characterise_uncertainty, vecvec_to_matrix

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

function vecvec_to_matrix(vecvec)
    """Function to turn vector of vector of floats to matrix for easier data access.
    
    Liberally stolen from stackoverflow (https://stackoverflow.com/questions/63892334/using-broadcasting-julia-for-converting-vector-of-vectors-to-matrices)
    
    """
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Float64, dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
end

# end