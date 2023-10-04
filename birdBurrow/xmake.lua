add_rules("mode.debug", "mode.release")
add_requires("glfw")
add_requires("wgpu-native", "glfw3webgpu")
add_requires("glm")
add_requires("stb")
set_policy("build.warning", true) -- show warnings
set_warnings("all") -- warn about many things

target("helloworld")
    set_kind("binary")
    set_languages("cxx20")

    add_deps("birdBurrow")
    
    add_files("demo/helloworld.cpp")
    set_rundir("$(projectdir)")

target("birdBurrow")
    set_kind("static")
    set_languages("cxx20")
    
    -- Declare our engine's header path.
    -- This allows targets that depend on the engine to #include them.
    add_includedirs("src", {public = true})
    add_packages("glfw", {public = true})
    add_packages( "wgpu-native", "glfw3webgpu",{public = true}) 
    add_packages("glm", {public = true})
    add_packages("stb")    
    -- Add all .cpp files in the `src` directory.
    add_files("src/*.cpp")



