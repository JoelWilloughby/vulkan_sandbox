use vulkano;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::Sdl;

use vulkano::instance::{InstanceExtensions, Instance, RawInstanceExtensions, layers_list, PhysicalDevice, debug::{DebugCallback, MessageType, MessageSeverity}};
use vulkano::device::{Device, Features, Queue, DeviceExtensions};
use vulkano::swapchain::{Surface, Capabilities, Swapchain, ColorSpace, CompositeAlpha, PresentMode, FullscreenExclusive, acquire_next_image};
use vulkano::format::{Format, FormatDesc, ClearValue};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::sync::{SharingMode, GpuFuture};

use std::ffi::CString;
use std::sync::Arc;
use sdl2::video::{Window};

use std::convert::From;
use std::collections::HashMap;
use std::cmp::{min, max};
use vulkano::VulkanObject;
use vulkano::framebuffer::{RenderPassAbstract, Subpass, Framebuffer, FramebufferAbstract};
use vulkano::pipeline::{GraphicsPipeline};
use vulkano::pipeline::vertex::{BufferlessDefinition, BufferlessVertices};
use vulkano::pipeline::viewport::Viewport;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SubpassContents, DynamicState, AutoCommandBuffer};
use vulkano::descriptor::PipelineLayoutAbstract;

const WIDTH: u32 = 1600;
const HEIGHT: u32 = 900;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_KHRONOS_validation",
];

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics: Option<u32>,
    presentable: Option<u32>,
}

type SdlVulkanSurface = Surface<()>;
type SdlVulkanImage = SwapchainImage<()>;
type SdlVulkanSwapchain = Swapchain<()>;
type ConcreteGraphicsPipeline = GraphicsPipeline<BufferlessDefinition, Box<dyn PipelineLayoutAbstract + Send + Sync + 'static>, Arc<dyn RenderPassAbstract + Send + Sync + 'static>>;

pub struct HelloTriangleApplication {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,
    surface: Arc<SdlVulkanSurface>,
    // Lifetime issues prevent storing the physical device directly, just look it up in the instance
    physical_device_index: usize,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
    swap_chain: Arc<SdlVulkanSwapchain>,
    // By default, vulkano gives us reasonable image views into the swap chain images accessible
    // via the SwapchainImage. vulkano's support for more image view configuration is currently
    // a little lacking, so we just use those
    swap_chain_images: Vec<Arc<SdlVulkanImage>>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<ConcreteGraphicsPipeline>,
    frame_buffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    // SDL2 stuff
    sdl_context: Sdl,
    window: Window,

    needs_recreate: bool,
}

impl QueueFamilyIndices {
    pub fn create(physical_device: &PhysicalDevice, surface: &Arc<SdlVulkanSurface>) -> Self {
        let mut indices = Self {
            graphics: None,
            presentable: None,
        };

        for queue_family in physical_device.queue_families() {
            if queue_family.supports_graphics() {
                indices.graphics = Some(queue_family.id());
            }
            if surface.is_supported(queue_family).unwrap_or(false) {
                indices.presentable = Some(queue_family.id());
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    pub fn is_complete(&self) -> bool {
        return self.graphics.is_some() && self.presentable.is_some();
    }
}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        let (sdl_context, window) = Self::init_window();
        let instance = Self::init_vulkan(&window);
        let debug_callback = Self::setup_debug_callback(&instance);
        let surface = Self::create_surface(instance.clone(), &window);
        let physical_device_index = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(&instance, &surface, physical_device_index);

        // Needs to be done each time the swap chain needs to be re-created
        let (swap_chain, swap_chain_images) = Self::create_swap_chain(&instance, &surface, device.clone(), physical_device_index, &window);
        let render_pass = Self::create_render_pass(device.clone(), &swap_chain.format());
        let graphics_pipeline = Self::create_graphics_pipeline(device.clone(), swap_chain.clone(), render_pass.clone());
        let frame_buffers = Self::create_frame_buffers(&swap_chain_images, render_pass.clone());

        let mut app = Self {
            instance,
            debug_callback,
            surface,
            physical_device_index,
            device,
            graphics_queue,
            present_queue,
            swap_chain,
            swap_chain_images,
            render_pass,
            graphics_pipeline,
            frame_buffers,
            sdl_context,
            window,

            // Create these after build to avoid lifetime issues
            command_buffers: vec![],
            needs_recreate: false,
        };

        app.create_command_buffers();
        app
    }

    fn recreate_swap_chain(&mut self) {
        let physical_device = PhysicalDevice::from_index(&self.instance, self.physical_device_index).unwrap();
        let capabilities = self.surface.capabilities(physical_device).unwrap();
        let dimensions = Self::choose_swap_extent(&capabilities, &self.window);

        let(swap_chain, swap_chain_images) = self.swap_chain.recreate_with_dimensions(dimensions).unwrap();
        let render_pass = Self::create_render_pass(self.device.clone(), &swap_chain.format());
        let graphics_pipeline = Self::create_graphics_pipeline(self.device.clone(), swap_chain.clone(), render_pass.clone());
        let frame_buffers = Self::create_frame_buffers(&swap_chain_images, render_pass.clone());

        self.swap_chain = swap_chain;
        self.swap_chain_images = swap_chain_images;
        self.render_pass = render_pass;
        self.graphics_pipeline = graphics_pipeline;
        self.frame_buffers = frame_buffers;

        self.create_command_buffers();

        self.needs_recreate = false;
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();
        self.command_buffers = self.frame_buffers.iter().map(|buffer| {
            let vertices = BufferlessVertices {
                vertices: 3, instances: 1,
            };

            let mut builder = AutoCommandBufferBuilder::primary_simultaneous_use(self.device.clone(), queue_family).unwrap();
            builder
                .begin_render_pass(buffer.clone(), SubpassContents::Inline, vec![ClearValue::Float([0.0, 0.0, 0.0, 1.0])]).unwrap()
                .draw(self.graphics_pipeline.clone(), &DynamicState::none(), vertices, (), ()).unwrap()
                .end_render_pass().unwrap();

            Arc::new(
                builder.build().unwrap()
            )
        }).collect();
    }

    fn create_frame_buffers(swap_chain_images: &Vec<Arc<SdlVulkanImage>>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>>{
        swap_chain_images.iter().map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone()).unwrap()
                    .build().unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect()
    }

    fn create_render_pass(device: Arc<Device>, color_format: &impl FormatDesc) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(vulkano::single_pass_renderpass! (
            device.clone(),
            attachments: {
                // Define the "color" attachment
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format.format(),
                    samples: 1,
                }
            },
            pass: {
                // Use the defined "color" attachment
                // The color attachment here being index 0 directly corresponds to the
                // layout(location = 0) out vec4 out_color in the frag shader.
                color: [color],
                depth_stencil: {}
            }
        ).unwrap())
    }

    fn create_graphics_pipeline(device: Arc<Device>, swap_chain: Arc<SdlVulkanSwapchain>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>) -> Arc<ConcreteGraphicsPipeline> {
        // We let the vulkano shader subsystem do the heavy lifting here.
        // These are compiled at build time into binary files and linked into
        // the final executable
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "./shaders/default.vert",
            }
        }
        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "./shaders/default.frag",
            }
        }

        let vs = vs::Shader::load(device.clone()).expect("failed to load vertex shader!");
        let fs = fs::Shader::load(device.clone()).expect("failed to load fragment shader!");

        let extent = swap_chain.dimensions();
        let viewport = vec![
            Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: [extent[0] as f32, extent[1] as f32]
            }
        ];

        Arc::new(GraphicsPipeline::start()
            .vertex_input(BufferlessDefinition {})
            .triangle_list()
            .primitive_restart(false)
            .viewports(viewport)
            .depth_clamp(false)
            .polygon_mode_fill()
            .line_width(1.0)
            .cull_mode_back()
            .front_face_clockwise()
            .sample_shading_disabled()
            .blend_pass_through()
            // Can be used to not have to recreate the pipeline each time the window resizes
//            .line_width_dynamic()
//            .viewports_scissors_dynamic(0)
            .vertex_shader(vs.main_entry_point(), ())
            .fragment_shader(fs.main_entry_point(), ())
            // One graphics pipeline is used for a single subpass of the render pass
            // If you wanted to do more effects on top, then you would need another render pass
            // and another graphics pipeline. Is this roughly equivalent to OpenGL's
            // linked program?
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone()).unwrap()
        )
    }

    fn choose_format(capabilities: &Capabilities) -> (impl FormatDesc, ColorSpace) {
        for format_desc in capabilities.supported_formats.iter() {
            if let (Format::R8G8B8A8Srgb, ColorSpace::SrgbNonLinear) = format_desc {
                return format_desc.clone();
            }
        }

        // Just us the first one...
        capabilities.supported_formats[0]
    }

    fn choose_present_mode(capabilities: &Capabilities) -> PresentMode {
        if capabilities.present_modes.mailbox {
            return PresentMode::Mailbox;
        }

        PresentMode::Fifo
    }

    fn choose_swap_extent(capabilities: &Capabilities, window: &Window) -> [u32; 2] {
        capabilities.current_extent.unwrap_or_else(|| {
            let (mut width, mut height) = window.drawable_size();
            width = max(width, capabilities.max_image_extent[0]);
            height = max(height, capabilities.max_image_extent[1]);
            [width, height]
        })
    }

    fn create_swap_chain(instance: &Arc<Instance>, surface: &Arc<SdlVulkanSurface>, device: Arc<Device>, physical_device_index: usize, window: &Window) -> (Arc<SdlVulkanSwapchain>, Vec<Arc<SdlVulkanImage>>){
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index).unwrap();
        let capabilities = surface.capabilities(physical_device).unwrap();
        let indices = QueueFamilyIndices::create(&physical_device, surface);

        let num_images = min(capabilities.min_image_count + 1, capabilities.max_image_count.unwrap_or(u32::MAX));
        let (format, color_space) = Self::choose_format(&capabilities);
        let dimensions = Self::choose_swap_extent(&capabilities, window);
        let indices = vec![indices.graphics.unwrap(), indices.presentable.unwrap()];
        let present_mode = Self::choose_present_mode(&capabilities);

        Swapchain::new(
            device,
            surface.clone(),
            num_images,
            format,
            dimensions,
            1u32, //layers
            ImageUsage::color_attachment(),
            if indices[0] == indices[1] {SharingMode::Exclusive} else {SharingMode::Concurrent(indices)},
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            FullscreenExclusive::Default,
            true, //clipped
            color_space
        ).expect("failed to create swap chain!")
    }

    fn create_surface(instance: Arc<Instance>, window: &Window) -> Arc<SdlVulkanSurface> {
        unsafe {
            let surface_raw = window.vulkan_create_surface(instance.internal_object()).expect("failed to create surface!");
            Arc::new(Surface::from_raw_surface(instance, surface_raw, ()))
        }
    }

    fn create_logical_device(instance: &Arc<Instance>, surface: &Arc<SdlVulkanSurface>, physical_device_index: usize) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index).unwrap();
        let indices = QueueFamilyIndices::create(&physical_device, surface);

        let features = Features::default();
        let mut queue_families = HashMap::new();
        queue_families.insert(indices.graphics.unwrap(), (physical_device.queue_family_by_id(indices.graphics.unwrap()).unwrap(), 1.0f32));
        queue_families.insert(indices.presentable.unwrap(), (physical_device.queue_family_by_id(indices.presentable.unwrap()).unwrap(), 1.0f32));
        let extensions = Self::required_device_extensions();

        let (device, queues_iter) = Device::new(
            physical_device, &features, &extensions, queue_families.values().cloned()
        ).expect("failed to create logical device!");

        let mut graphics_queue = None;
        let mut present_queue = None;
        for queue in queues_iter {
            if queue.family().id() == indices.graphics.unwrap() {
                graphics_queue = Some(queue.clone());
            }
            if queue.family().id() == indices.presentable.unwrap() {
                present_queue = Some(queue.clone());
            }
        }

        (device, graphics_queue.unwrap(), present_queue.unwrap())
    }

    fn required_device_extensions() -> DeviceExtensions {
        let mut ext = DeviceExtensions::none();
        ext.khr_swapchain = true;
        ext
    }

    fn check_device_extensions_support(device: &PhysicalDevice) -> bool {
        let supported_extensions = DeviceExtensions::supported_by_device(*device);
        Self::required_device_extensions().difference(&supported_extensions) == DeviceExtensions::none()
    }

    fn is_device_suitable(device: &PhysicalDevice, surface: &Arc<SdlVulkanSurface>) -> bool {
        let indices = QueueFamilyIndices::create(device, surface);
        if !indices.is_complete() {
            return false;
        }
        if !Self::check_device_extensions_support(device) {
            return false;
        }

        // Check capabilities of the swap chain
        let capabilites = surface.capabilities(*device).expect("failed to get capabilities");
        if capabilites.supported_formats.is_empty() {
            return false;
        }
        if capabilites.present_modes.iter().next().is_none() {
            return false;
        }

        true
    }

    fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<SdlVulkanSurface>) -> usize {
        for (index, physical_device) in PhysicalDevice::enumerate(&instance).enumerate() {
            if Self::is_device_suitable(&physical_device, surface) {
                // Just return the first thing you find
                return index;
            }
        }

        panic!("No suitable device found");
    }

    fn check_validation_layer_support() -> bool {
        let layers : Vec<_> = layers_list().unwrap().map(|layer| layer.name().to_owned()).collect();
        VALIDATION_LAYERS.iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let severity = MessageSeverity {
            error: true,
            information: false,
            verbose: false,
            warning: true,
        };

        let message_type = MessageType {
            general: true,
            performance: true,
            validation: true,
        };

        DebugCallback::new(&instance, severity, message_type, |msg| {
            println!("validation layer: {:?}", msg.description);
        }).ok()
    }

    // Returns the extensions required by this application
    // This includes those extensions that are necessary for the passed in window
    // object as well as any debug extensions needed
    fn get_required_extensions(window: &Window) -> RawInstanceExtensions {
        let extensions = window.vulkan_instance_extensions()
            .expect("failed to load vulkan extensions for sdl2 window");

        let mut extensions = RawInstanceExtensions::new(
            extensions.iter().map(
                |&v| CString::new(v).unwrap()
            )
        );

        if ENABLE_VALIDATION_LAYERS {
            let mut ext = InstanceExtensions::none();
            ext.ext_debug_utils = true;
            extensions = extensions.union(&RawInstanceExtensions::from(&ext));
        }

        extensions
    }

    // Initializes the vulkan subsystem and returns a pointer to the initialized
    // Vulkan instance
    fn init_vulkan(window: &sdl2::video::Window) -> Arc<Instance> {
        // Check for validation layer support
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            println!("Validation layers requested, but none found!");
        }

        // Let the user know what extensions can be supported
        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions");
        println!("Supported extensions: {:?}", supported_extensions);

        let application_info =  vulkano::app_info_from_cargo_toml!();

        let _instance_extensions = window.vulkan_instance_extensions()
            .expect("failed to load vulkan extensions for sdl2 window");

        let required_extensions = Self::get_required_extensions(window);

        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(Some(&application_info), required_extensions, VALIDATION_LAYERS.iter().cloned())
                .expect("failed to create Vulkan instance")
        } else {
            Instance::new(Some(&application_info), required_extensions, None)
                .expect("failed to create Vulkan instance")
        }
    }

    // Initializes the SDL window and video subsystem
    fn init_window() -> (Sdl, Window) {
        let sdl_context = sdl2::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();

        let window = video_subsystem.window("Vulkan Sandbox", WIDTH, HEIGHT)
            .resizable()
            .vulkan()
            .build()
            .unwrap();

        (sdl_context, window)
    }

    fn draw_frame(&mut self) {
        let (idx, sub_optimal, acquire_future) = acquire_next_image(self.swap_chain.clone(), None).unwrap();
        let command_buffer = self.command_buffers[idx].clone();
        let future = acquire_future
                .then_execute(self.graphics_queue.clone(), command_buffer).unwrap()
                .then_swapchain_present(self.present_queue.clone(), self.swap_chain.clone(), idx)
                .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                future.wait(None).unwrap();
            },
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.needs_recreate = true;
            },
            Err(e) => {
                println!("Failed to flush future {:?}", e);
            }
        }

        if sub_optimal {
            self.needs_recreate = true;
        }
    }

    fn main_loop(&mut self) {
        let mut done = false;
        let mut event_pump = self.sdl_context.event_pump().unwrap();
        while !done {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit {..} | Event::KeyDown {keycode: Some(Keycode::Escape), .. } => {
                        done = true;
                    },
                    Event::Window {win_event: event, ..} => {
                        match event {
                            sdl2::event::WindowEvent::Resized(..) | sdl2::event::WindowEvent::SizeChanged(..) => {
                                self.needs_recreate = true;
                            },
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }

            // Draw shit
            self.draw_frame();

            if self.needs_recreate {
                self.recreate_swap_chain();
            }

            ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
        }

    }
}

fn main() {
    let mut app = HelloTriangleApplication::initialize();
    app.main_loop();
}
