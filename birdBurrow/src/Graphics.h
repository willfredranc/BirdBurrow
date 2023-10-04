#pragma once
#include "GLFW/glfw3.h"
#include <string>
#include <vector>
#include "glm/glm.hpp"

using namespace glm;

namespace birdBurrow
{
    struct Sprite{
        //image name
        std::string name;
        //position
        vec3 translation;
        //scale
        vec2 scale;
        //rotation
        vec4 rotation;
        //z value
        float z;
        //texture

    };

    class Graphics
    {
        public :
            //Vars
            GLFWwindow* window;
            bool window_fullscreen;
            int window_w;
            int window_h;
            const char* window_name;
            //Functions
            Graphics();
            Graphics(int w, int h, bool fullscreen);
            void Startup();
            void Shutdown();
            void Update();
            void CreatePipeline();
            void Draw(const std::vector<Sprite>& sprites); 
            bool LoadTexture( const std::string &name, const std::string &path );
            
    };
    inline Graphics gGraphics;
}
