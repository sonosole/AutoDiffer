function readtype(suffix::String,path::String)
    typefiles = Vector{String}(undef, 0)
    for file in readdir(path)
        if suffix in splitext(file)
            push!(typefiles,file)
        end
    end
    return typefiles
end


function rmtype(suffix::String,path::String;force=false,details=false)
    for file in readdir(path)
        if suffix in splitext(file)
            filelink = joinpath(path,file)
            rm(filelink;force=force)
            if details
                println("removing $(filelink)")
            end
        end
    end
end


function cptype(suffix::String,src::String,dst::String;force=false,details=false)
    for file in readdir(src)
        if suffix in splitext(file)
            srclink = joinpath(src,file)
            dstlink = joinpath(dst,file)
            cp(srclink,dstlink,force=force)
            if details
                println("copying $(file) from $(src) to $(dst)")
            end
        end
    end
end


function mvtype(suffix::String,src::String,dst::String;force=false,details=false)
    for file in readdir(src)
        if suffix in splitext(file)
            srclink = joinpath(src,file)
            dstlink = joinpath(dst,file)
            mv(srclink,dstlink,force=force)
            if details
                println("moving $(file) from $(src) to $(dst)")
            end
        end
    end
end
