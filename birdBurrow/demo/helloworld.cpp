#include <iostream>
#include "../src/Engine.h"

using namespace birdBurrow;

int main( int argc, const char* argv[]) {

    gEngine.Startup();
    gEngine.RunGameLoop([&]{gEngine.Update();});
    gEngine.Shutdown();
    return 0;
}
