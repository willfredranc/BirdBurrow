#pragma once
#include "Graphics.h"

namespace birdBurrow
{
    class Input
    {
        public:
            //vars
            GLFWwindow* window;

            //functions
            Input();
            void Startup();
            void Shutdown();
            void Update();
            bool KeyIsPressed(int key);
            bool KeyIsReleased(int key);

    };
    inline Input gInput;
}
