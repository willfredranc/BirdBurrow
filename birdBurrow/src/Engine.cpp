#include "Engine.h"
#include <thread>
#include <chrono>

namespace birdBurrow
{
    Engine::Engine()
    {
        //Graphics graphics();
    }

    struct Engine::config
    {

    };

    void Engine::Update()
    {
        gGraphics.Update();
        gInput.Update();
        //printf("Hello\n");
    }

    void Engine::Startup()
    {
        gGraphics.Startup();
        gInput.Startup();
    }

    void Engine::Shutdown()
    {
        gGraphics.Shutdown();
        gInput.Shutdown();
    }

    void Engine::RunGameLoop(std::function<void()> callback)
    {
        const auto sleepTime = std::chrono::duration<double>(1./60.);

        while(!glfwWindowShouldClose(gGraphics.window))
        {
            callback();
            std::this_thread::sleep_for(sleepTime);
        }
    }
}
