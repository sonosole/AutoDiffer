function DistanceMatrix(s::String, t::String)
    m = length(s)
    n = length(t)
    d = zeros(Int32, m + 1, n + 1)
    # source prefixes can be transformed into empty string by dropping all characters
    for i = 0:m
        d[i + 1, 1] = i
    end
    # target prefixes can be reached from empty source prefix by inserting every character
    for j = 0:n
        d[1, j + 1] = j
    end
   # Fill in distance matrix
    for j = 2:n + 1
        for i = 2:m + 1
            if s[i-1] == t[j-1]
                d[i,j] = d[i-1,j-1]
            else
                d[i,j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
            end
            #=
            d[i, j] := minimum(d[i-1, j] + 1,    // deletion -> 0
                               d[i, j-1] + 1,    // insertion -> 1
                               d[i-1, j-1] + 1)  // substitution -> 2
            =#
        end
    end
    return d
end

function editDist(s::String, t::String)
    array = DistanceMatrix(s, t);
    return array[length(s)+1,length(t)+1]
end

function showDist(s::String, t::String)
    array = DistanceMatrix(s, t)
    for i = 1:length(s)+1
        for j = 1:length(t)+1
            print(array[i,j]," ")
        end
        println()
    end
end

# print( "the distance from \33[33m",ARGS[1],"\e[0m to \33[33m",ARGS[2],"\e[0m is : ", editDist(ARGS[1],ARGS[2]) )
# print( "\n" )
# showDist(ARGS[1],ARGS[2])

showDist("abc","abcd")
editDist("a b c","abcdedf")
