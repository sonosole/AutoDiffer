function readtype(suffix::String, path::String)
    # read all files having a certain suffix like txt in a path
    typefiles = Vector{String}(undef, 0)
    for file in readdir(path)
        if suffix in splitext(file)
            push!(typefiles,file)
        end
    end
    return typefiles
end


function rmtype(suffix::String, path::String; force=false, details=false)
    # remove all files having a certain suffix like .wav .txt in a path
    c = 0
    for file in readdir(path)
        if suffix in splitext(file)
            filelink = joinpath(path,file)
            rm(filelink;force=force)
            if details
                c += 1
                println("removing [$c] $(filelink)")
            end
        end
    end
end


function cptype(suffix::String, src::String, dst::String; force=false, details=false)
    # copy all files having a certain suffix like .txt from src path to dst path
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


function mvtype(suffix::String, src::String, dst::String; force=false, details=false)
    # move all files having a certain suffix like .txt from src path to dst path
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


function mvall21dir(suffix::String, src::String, dest::String; force=false, details=false)
    # move all files having a certain suffix like .txt from src dir to dst dir
    c = 0
    for (root, dirs, files) in walkdir(src)
        for file in files
            if suffix in splitext(file)
                srclink = joinpath(root,file)
                dstlink = joinpath(dest,file)
                mv(srclink,dstlink,force=force)
                if details
                    c += 1
                    println("[$(c)] moving $(file) from $(src) to $(dest)")
                end
            end
        end
    end
end
