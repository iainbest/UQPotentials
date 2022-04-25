# module UtilityFunctions

# using Statistics

export characterise_uncertainty, vecvec_to_matrix
export virial_matrix_to_vector
export unpack_vecvec_of_3tuples

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

"""
    virial_matrix_to_vector(matrix)

Take 3x3 virial stress matrix, return corresponding voigt vector. In matrix notation (row,column), the 
elements 11,22,33,23,13,12 are contained in the vector, in that order.
"""
function virial_matrix_to_vector(matrix::AbstractArray{Float64})
    
    @assert size(matrix) == (3,3)
    
    m_vector = vec(transpose(matrix))
    
    voigt_indices = [1,5,9,6,3,2]
    
    voigt_vector = m_vector[voigt_indices]
    
    return voigt_vector
end

"""
    unpack_vecvec_of_3tuples(vecvec_3tuples)

Unpack vector of vector of tuples (with each tuple having 3 entries) in 3 vector of vectors of individual 
entries

### Arguments

- `vecvec_3tuples`
"""
function unpack_vecvec_of_3tuples(vecvec_3tuples)

    vecvec_tuple1 = Vector{Vector{Float64}}(undef,length(vecvec_3tuples))
    vecvec_tuple2 = Vector{Vector{Float64}}(undef,length(vecvec_3tuples))
    vecvec_tuple3 = Vector{Vector{Float64}}(undef,length(vecvec_3tuples))


    for (count_,tuple) in enumerate(vecvec_3tuples) 
        vec_tuple1 = zeros(length(tuple))
        vec_tuple2 = zeros(length(tuple))
        vec_tuple3 = zeros(length(tuple))
        for (count,i) in enumerate(tuple)

            vec_tuple1[count] = i[1]
            vec_tuple2[count] = i[2]
            vec_tuple3[count] = i[3]

        end

        vecvec_tuple1[count_] = vec_tuple1
        vecvec_tuple2[count_] = vec_tuple2
        vecvec_tuple3[count_] = vec_tuple3

    end
    
    return vecvec_tuple1,vecvec_tuple2,vecvec_tuple3
end

# end