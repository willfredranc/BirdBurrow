#include "Input.h"
#include <iostream>
namespace birdBurrow
{
    GLFWwindow* window;
    // callback for key input
    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        std::cout << key << std::endl;
    }

    Input::Input()
    {
    
    }
    void Input::Startup()
    {
        window = gGraphics.window;
        glfwSetKeyCallback(window, keyCallback);
    }
    void Input::Shutdown()
    {
    
    }
    void Input::Update()
    {
        glfwPollEvents();
    }
     
    //Checks for key release or key press
    //GLFW_RELEASE = 0
    //GLFW_PRESS = 1
    //GLFW_REPEAT = 2
    bool KeyIsPressed(int key)
    {
        if(glfwGetKey(window, key) == 1)
        {
            return true;
        }
        return false;
    } 
    bool KeyIsReleased(int key)
    {
        if(glfwGetKey(window, key) == 2)
        {
            return true;
        }
        return false;
    }
}
