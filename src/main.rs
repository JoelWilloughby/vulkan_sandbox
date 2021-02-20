use vulkano;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use vulkano::instance::{ApplicationInfo, Version, InstanceExtensions, Instance, RawInstanceExtensions, LayerProperties, layers_list, PhysicalDevice, QueueFamily};
use std::ffi::CString;
use std::sync::Arc;
use sdl2::video::{Window, WindowContext};

use std::convert::From;
use vulkano::instance::debug::{DebugCallback, MessageType, MessageSeverity};
use sdl2::Sdl;
use vulkano::VulkanObject;
use vulkano::device::{Device, RawDeviceExtensions, Features, Queue};
use vulkano::swapchain::Surface;
use std::rc::Rc;
use std::collections::HashMap;

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

type SdlVulkanSurface = Surface<Rc<WindowContext>>;

struct HelloTriangleApplication {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,
    surface: Arc<SdlVulkanSurface>,
    // Lifetime issues prevent storing the physical device directly, just look it up in the instance
    physical_device_index: usize,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    // SDL2 stuff
    sdl_context: Sdl,
    window: Window,
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

        Self {
            instance,
            debug_callback,
            surface,
            physical_device_index,
            device,
            graphics_queue,
            present_queue,
            sdl_context,
            window
        }
    }

    fn create_surface(instance: Arc<Instance>, window: &Window) -> Arc<SdlVulkanSurface> {
        let surface_raw = window.vulkan_create_surface(instance.internal_object()).expect("failed to create surface!");
        unsafe { Arc::new(Surface::from_raw_surface(instance, surface_raw, window.context())) }
    }

    fn create_logical_device(instance: &Arc<Instance>, surface: &Arc<SdlVulkanSurface>, physical_device_index: usize) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index).unwrap();
        let indices = QueueFamilyIndices::create(&physical_device, surface);

        let features = Features::default();
        let mut queue_families = HashMap::new();
        queue_families.insert(indices.graphics.unwrap(), (physical_device.queue_family_by_id(indices.graphics.unwrap()).unwrap(), 1.0f32));
        queue_families.insert(indices.presentable.unwrap(), (physical_device.queue_family_by_id(indices.presentable.unwrap()).unwrap(), 1.0f32));
        let extensions = RawDeviceExtensions::none();

        let (device, queues_iter) = Device::new(
            physical_device, &features, extensions, queue_families.values().cloned()
        ).expect("failed to create logical device!");

        // TODO how to get which queue is which ???
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

    fn is_device_suitable(device: &PhysicalDevice, surface: &Arc<SdlVulkanSurface>) -> bool {
        let indices = QueueFamilyIndices::create(device, surface);
        indices.is_complete()
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
            let mut debug_ext = InstanceExtensions::none();
            debug_ext.ext_debug_utils = true;
            extensions = extensions.union(&RawInstanceExtensions::from(&debug_ext));
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

        let application_info = ApplicationInfo {
            application_name: Some("Vulkan Sandbox".into()),
            application_version: Some(Version{major: 1, minor: 0, patch: 0}),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version{major: 1, minor: 0, patch: 0}),
        };

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
            .vulkan()
            .build()
            .unwrap();

        (sdl_context, window)
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
                    _ => {}
                }
            }

            ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
        }

    }
}

fn main() {
    let mut app = HelloTriangleApplication::initialize();
    app.main_loop();
}
