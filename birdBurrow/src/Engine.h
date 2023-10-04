#pragma once

#include "Graphics.h"
#include "Input.h"
#include <functional>

namespace birdBurrow
{
    class Engine{
        public:
            
            static Graphics graphics;

            Engine();
            void Startup();
            void Shutdown();
            void RunGameLoop(std::function<void()> callback);
            void Update();

            struct config;
    };

    inline Engine gEngine;
}
