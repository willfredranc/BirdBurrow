#define GLFW_INCLUDE_NONE#
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "GLFW/glfw3.h"
#include "Graphics.h"
#include <webgpu/webgpu.h>
#include <glfw3webgpu.h>
#include "glm/glm.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <complex>
#include <list>

template< typename T > constexpr const T* to_ptr( const T& val ) { return &val; }
template< typename T, std::size_t N > constexpr const T* to_ptr( const T (&&arr)[N] ) { return arr; }

using namespace glm;

namespace
{
    //Instance data fro manaing translations, rotations, etc. of
    //rectangles we are rendering
    struct InstanceData {
        vec3 translation;
        vec2 scale;
        vec4 rotation;
        float z;
    };
}

//make struct for image

namespace birdBurrow {

    //Instances for WebGPU
    WGPUInstance instance;
    WGPUSurface surface;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue _queue;

    WGPUBuffer vertex_buffer;
    WGPUTextureFormat swap_chain_format;
    WGPUSwapChain swapchain;
    WGPUBuffer uniform_buffer;
    WGPUSampler textSampler;
    WGPUShaderModuleWGSLDescriptor code_desc;
    WGPUShaderModuleDescriptor shader_desc;
    WGPUShaderModule shader_module;
    WGPURenderPipeline pipeline;
    WGPUTexture tex;
    WGPUBuffer instance_buffer;

    WGPUCommandEncoder encoder;
    WGPUTextureView current_texture_view;
    WGPURenderPassEncoder render_pass;

    const char* source = R"(struct Uniforms {
    projection: mat4x4f,
    };

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var texSampler: sampler;
    @group(0) @binding(2) var texData: texture_2d<f32>;

    struct VertexInput {
        @location(0) position: vec2f,
        @location(1) texcoords: vec2f,
        @location(2) translation: vec3f,
        @location(3) scale: f32,
    };

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) texcoords: vec2f,
    };

    @vertex
    fn vertex_shader_main( in: VertexInput ) -> VertexOutput {
        var out: VertexOutput;
        out.position = uniforms.projection * vec4f( vec3f( in.scale * in.position, 0.0 ) + in.translation, 1.0 );
        out.texcoords = in.texcoords;
        return out;
    }

    @fragment
    fn fragment_shader_main( in: VertexOutput ) -> @location(0) vec4f {
        let color = textureSample( texData, texSampler, in.texcoords ).rgba;
        return color;
    })";


    // A vertex buffer containing a textured square.
    const struct {
        // position
        float x, y;
        // texcoords
        float u, v;
    } vertices[] = {
          // position       // texcoords
        { -1.0f,  -1.0f,    0.0f,  1.0f },
        {  1.0f,  -1.0f,    1.0f,  1.0f },
        { -1.0f,   1.0f,    0.0f,  0.0f },
        {  1.0f,   1.0f,    1.0f,  0.0f },
    };

    //Uniform Struct
    struct Uniforms {
        mat4 projection;
        Uniforms(mat4 projection) : projection(projection){}
        ~Uniforms(){}
    };

    //Image Struct
    struct Image {
        int width, height;
        WGPUTexture texture;

        /*Image(int w, int h, WGPUTexture t)
        {
            width = w;
            height = h;
            texture = t;
        }*/

        ~Image()
        {
            if(texture)
            {
                wgpuTextureRelease(texture);
            }
            else
            {
                wgpuTextureDestroy(texture);
            }
        }
    };

    std::unordered_map<std::string, Image> imageMap{};

    Graphics::Graphics()
    {
        window_fullscreen = false;
        window_w = 400;
        window_h = 400;
        window_name = "BirdBurrow";

    }

    Graphics::Graphics(int w, int h, bool fullscreen)
    {
        window_fullscreen = fullscreen;
        window_w = w;
        window_h = h;
        window_name = "BirdBurrow";
    }

    void Graphics::Startup(){
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        //create window
        window = glfwCreateWindow( window_w, window_h, window_name, window_fullscreen ? glfwGetPrimaryMonitor() : 0, 0);
        glfwSetWindowAspectRatio( window, window_w, window_h);
        if(!window)
        {
            std::cerr << "Failed to create window" << std::endl;
            glfwTerminate();
        }

        //Start of initializing WebGPU
        /***INSTANCE***/
        instance = wgpuCreateInstance( to_ptr( WGPUInstanceDescriptor{} ) );
        if(!instance)
        {
            std::cerr << "Failed to initialize WebGPU" << std::endl;
            glfwTerminate();
        }

        //initialize rest of the sequence
        /***SURFACE***/
        surface = glfwGetWGPUSurface( instance, window );

        /***ADAPTER***/
        adapter = nullptr;
        wgpuInstanceRequestAdapter(
            instance,
            to_ptr( WGPURequestAdapterOptions{ .compatibleSurface = surface, .backendType = WGPUBackendType_OpenGLES  } ),
            []( WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* adapter_ptr ) {
                if( status != WGPURequestAdapterStatus_Success ) {
                    std::cerr << "Failed to get a WebGPU adapter: " << message << std::endl;
                    glfwTerminate();
                }

                *static_cast<WGPUAdapter*>(adapter_ptr) = adapter;
            },
            &(adapter)
            );
        /***DEVICE***/
        WGPUDevice device = nullptr;
        wgpuAdapterRequestDevice(
            adapter,
            nullptr,
            []( WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* device_ptr ) {
                if( status != WGPURequestDeviceStatus_Success ) {
                    std::cerr << "Failed to get a WebGPU device: " << message << std::endl;
                    glfwTerminate();
                }

                *static_cast<WGPUDevice*>(device_ptr) = device;
            },
            &(device)
        );

        // An error callback to help with debugging
        wgpuDeviceSetUncapturedErrorCallback(
            device,
            []( WGPUErrorType type, char const* message, void* ) {
                std::cerr << "WebGPU uncaptured error type " << type << " with message: " << message << std::endl;
            },
            nullptr
            );
        /***QUEUE***/
        _queue = wgpuDeviceGetQueue( device );

        /***VERTEX_BUFFER***/
        vertex_buffer = wgpuDeviceCreateBuffer( device, to_ptr( WGPUBufferDescriptor{
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
            .size = sizeof(vertices)
            }) );
        wgpuQueueWriteBuffer( _queue, vertex_buffer, 0, vertices, sizeof(vertices) );

        /*********SWAP CHAIN****************/
        swap_chain_format = wgpuSurfaceGetPreferredFormat( surface, adapter );

        glfwGetFramebufferSize( window, &window_w, &window_h );
        swapchain = wgpuDeviceCreateSwapChain( device, surface, to_ptr( WGPUSwapChainDescriptor{
            .usage = WGPUTextureUsage_RenderAttachment,
            .format = swap_chain_format,
            .width = (uint32_t)window_w,
            .height = (uint32_t)window_h
            }) );

        /***UNIFORM_BUFFER***/
        uniform_buffer = wgpuDeviceCreateBuffer( device, to_ptr( WGPUBufferDescriptor{
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
            .size = sizeof(Uniforms)
            }) );

        /***TEXT_SAMPLER***/
        textSampler = wgpuDeviceCreateSampler( device, to_ptr( WGPUSamplerDescriptor{
            .addressModeU = WGPUAddressMode_ClampToEdge,
            .addressModeV = WGPUAddressMode_ClampToEdge,
            .magFilter = WGPUFilterMode_Linear,
            .minFilter = WGPUFilterMode_Linear,
            .maxAnisotropy = 1
            } ) );

        /***SHADER_MODULE***/
        code_desc = {};
        code_desc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
        code_desc.code = source; // The shader source as a `char*`
        shader_desc = {};
        shader_desc.nextInChain = &code_desc.chain;
        shader_module = wgpuDeviceCreateShaderModule( device, &shader_desc );

        /***PIPELINE***/
        Graphics::CreatePipeline();
    }

    void Graphics::Shutdown()
    {
        imageMap.clear();
        wgpuShaderModuleRelease(shader_module);
        wgpuSamplerRelease(textSampler);
        wgpuBufferRelease(uniform_buffer);
        wgpuSwapChainRelease(swapchain);
        wgpuBufferRelease(vertex_buffer);
        wgpuQueueRelease(_queue);
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuSurfaceRelease(surface);
        wgpuInstanceRelease(instance);
        glfwTerminate();
    }
    void Graphics::Update()
    {

    }

    bool Graphics::LoadTexture( const std::string& name, const std::string& path )
    {
        int width, height, channels;
        unsigned char* data = stbi_load( path.c_str(), &width, &height, &channels, 4 );
        tex = wgpuDeviceCreateTexture( device, to_ptr( WGPUTextureDescriptor{
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .dimension = WGPUTextureDimension_2D,
            .size = { (uint32_t)width, (uint32_t)height, 1 },
            .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1,
            .sampleCount = 1
            } ) );

        wgpuQueueWriteTexture(
            _queue,
            to_ptr<WGPUImageCopyTexture>({ .texture = tex }),
            data,
            width * height * 4,
            to_ptr<WGPUTextureDataLayout>({ .bytesPerRow = (uint32_t)(width*4), .rowsPerImage = (uint32_t)height }),
            to_ptr( WGPUExtent3D{ (uint32_t)width, (uint32_t)height, 1 } )
            );
        //free data
        stbi_image_free(data);
        imageMap[name] = {.width = width, .height = height, .texture = tex};
        return true;
    }

    void Graphics::CreatePipeline()
    {
        pipeline = wgpuDeviceCreateRenderPipeline( device, to_ptr( WGPURenderPipelineDescriptor{

        // Describe the vertex shader inputs
        .vertex = {
            .module = shader_module,
            .entryPoint = "vertex_shader_main",
            // Vertex attributes.
            .bufferCount = 2,
            .buffers = to_ptr<WGPUVertexBufferLayout>({
                // We have one buffer with our per-vertex position and UV data. This data never changes.
                // Note how the type, byte offset, and stride (bytes between elements) exactly matches our `vertex_buffer`.
                {
                    .arrayStride = 4*sizeof(float),
                    .attributeCount = 2,
                    .attributes = to_ptr<WGPUVertexAttribute>({
                        // Position x,y are first.
                        {
                            .format = WGPUVertexFormat_Float32x2,
                            .offset = 0,
                            .shaderLocation = 0
                        },
                        // Texture coordinates u,v are second.
                        {
                            .format = WGPUVertexFormat_Float32x2,
                            .offset = 2*sizeof(float),
                            .shaderLocation = 1
                        }
                        })
                },
                // We will use a second buffer with our per-sprite translation and scale. This data will be set in our draw function.
                {
                    .arrayStride = sizeof(InstanceData),
                    // This data is per-instance. All four vertices will get the same value. Each instance of drawing the vertices will get a different value.
                    // The type, byte offset, and stride (bytes between elements) exactly match the array of `InstanceData` structs we will upload in our draw function.
                    .stepMode = WGPUVertexStepMode_Instance,
                    .attributeCount = 2,
                    .attributes = to_ptr<WGPUVertexAttribute>({
                        // Translation as a 3D vector.
                        {
                            .format = WGPUVertexFormat_Float32x3,
                            .offset = offsetof(InstanceData, translation),
                            .shaderLocation = 2
                        },
                        // Scale as a 2D vector for non-uniform scaling.
                        {
                            .format = WGPUVertexFormat_Float32x2,
                            .offset = offsetof(InstanceData, scale),
                            .shaderLocation = 3
                        }
                        })
                }
                })
            },

        // Interpret our 4 vertices as a triangle strip
        .primitive = WGPUPrimitiveState{
            .topology = WGPUPrimitiveTopology_TriangleStrip,
            },

        // No multi-sampling (1 sample per pixel, all bits on).
        .multisample = WGPUMultisampleState{
            .count = 1,
            .mask = ~0u
            },

        // Describe the fragment shader and its output
        .fragment = to_ptr( WGPUFragmentState{
            .module = shader_module,
            .entryPoint = "fragment_shader_main",

            // Our fragment shader outputs a single color value per pixel.
            .targetCount = 1,
            .targets = to_ptr<WGPUColorTargetState>({
                {
                    .format = swap_chain_format,
                    // The images we want to draw may have transparency, so let's turn on alpha blending with over compositing (ɑ⋅foreground + (1-ɑ)⋅background).
                    // This will blend with whatever has already been drawn.
                    .blend = to_ptr( WGPUBlendState{
                        // Over blending for color
                        .color = {
                            .operation = WGPUBlendOperation_Add,
                            .srcFactor = WGPUBlendFactor_SrcAlpha,
                            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha
                            },
                        // Leave destination alpha alone
                        .alpha = {
                            .operation = WGPUBlendOperation_Add,
                            .srcFactor = WGPUBlendFactor_Zero,
                            .dstFactor = WGPUBlendFactor_One
                            }
                        } ),
                    .writeMask = WGPUColorWriteMask_All
                }})
            } )
        } ) );
    }

    //color code for background
    float red = 1.0;
    float blue = 1.0;
    float green =  1.0;
    void Graphics::Draw(const std::vector<Sprite>& sprites)
    {
        // auto copy of sprites
        auto copy = sprites;
        Sprite sprite;
        int image_width, image_height;

        //allocate buffer for InstanceData
        instance_buffer = wgpuDeviceCreateBuffer( device, to_ptr<WGPUBufferDescriptor>({
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
            .size = sizeof(InstanceData) * sprites.size()
            }) );

        //create command encoder
        encoder = wgpuDeviceCreateCommandEncoder( device, nullptr );
        // get window's swapchain's texture view
        current_texture_view = wgpuSwapChainGetCurrentTextureView( swapchain );
        //begin our render pass by clearing the screen;
        render_pass = wgpuCommandEncoderBeginRenderPass( encoder, to_ptr<WGPURenderPassDescriptor>({
            .colorAttachmentCount = 1,
            .colorAttachments = to_ptr<WGPURenderPassColorAttachment>({{
                .view = current_texture_view,
                .loadOp = WGPULoadOp_Clear,
                .storeOp = WGPUStoreOp_Store,
                // Choose the background color.
                .clearValue = WGPUColor{ red, green, blue, 1.0 }
                }})
            }) );
        //set pipeline
        wgpuRenderPassEncoderSetPipeline( render_pass, pipeline );
        //attach vertex data for our quad as slot 0
        wgpuRenderPassEncoderSetVertexBuffer( render_pass, 0 /* slot */, vertex_buffer, 0, 4*4*sizeof(float) );
        //attach the per-sprite instance data as slot 1
        wgpuRenderPassEncoderSetVertexBuffer( render_pass, 1 /* slot */, instance_buffer, 0, sizeof(InstanceData) * sprites.size() );

        //****Make a Uniforms struct -> copy to GPU ****/
        // Start with an identity matrix.
        struct Uniforms uniforms = {mat4{1}};
        // Scale x and y by 1/100.

        uniforms.projection[0][0] = uniforms.projection[1][1] = 1./100.;
        // Scale the long edge by an additional 1/(long/short) = short/long.
        if( window_w < window_h ) {
            uniforms.projection[1][1] *= window_w;
            uniforms.projection[1][1] /= window_h;
        } else {
            uniforms.projection[0][0] *= window_h;
            uniforms.projection[0][0] /= window_w;
        }
        wgpuQueueWriteBuffer(_queue, uniform_buffer, 0, &uniforms, sizeof(Uniforms));

        /***SORT sprites so we can draw them back-to-front (z values)***/
        std::sort(copy.begin(), copy.end(), [](const Sprite& lhs, const Sprite& rhs) {return lhs.z > rhs.z;} );

        /****DRAW each sprite***/
        //scale -> rotate -> translate
        //auto layout = wgpuRenderPipelineGetBindGroupLayout( pipeline, 0 );

        for(int i=0; i < sprites.size(); i++)
        {
            InstanceData data{};
            sprite = copy[i];
            data.translation = sprite.translation;
            data.scale = sprite.scale;
            data.rotation = sprite.rotation;
            data.z = sprite.z;

            image_width = imageMap[sprite.name].width;
            image_height = imageMap[sprite.name].height;
            if( image_width < image_height ) {
                data.scale *= vec2( std::real(image_width)/image_height, 1.0 );
            } else {
                data.scale *= vec2( 1.0, std::real(image_height)/image_width );
            }
            wgpuQueueWriteBuffer( _queue, instance_buffer, i * sizeof(InstanceData), &data, sizeof(InstanceData) );

            /***Bind groups***/
            auto layout = wgpuRenderPipelineGetBindGroupLayout( pipeline, 0 );
            WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup( device, to_ptr( WGPUBindGroupDescriptor{
                .layout = layout,
                .entryCount = 3,
                // The entries `.binding` matches what we wrote in the shader.
                .entries = to_ptr<WGPUBindGroupEntry>({
                    {
                        .binding = 0,
                        .buffer = uniform_buffer,
                        .size = sizeof( Uniforms )
                    },
                    {
                        .binding = 1,
                        .sampler = textSampler,
                    },
                    {
                        .binding = 2,
                        .textureView = wgpuTextureCreateView( imageMap[sprite.name].texture, nullptr )
                    }
                    })
                } ) );
            //wgpuBindGroupLayoutRelease( layout );
            wgpuRenderPassEncoderSetBindGroup( render_pass, 0, bind_group, 0, nullptr );
            wgpuRenderPassEncoderDraw(render_pass, 4, 1, 0, i);
        }
                //finish drawing
        wgpuRenderPassEncoderEnd( render_pass );
        WGPUCommandBuffer command = wgpuCommandEncoderFinish( encoder, nullptr );
        wgpuQueueSubmit( _queue, 1, &command );
        wgpuSwapChainPresent( swapchain );
        //free buffer
        wgpuBufferRelease(instance_buffer);

    }

}

