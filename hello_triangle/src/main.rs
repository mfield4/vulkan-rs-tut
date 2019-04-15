extern crate sdl2;
extern crate vulkano;
extern crate vulkano_win;

use std::collections::HashSet;
use std::iter::FromIterator;
use std::sync::Arc;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::video::{Window, WindowBuilder};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::instance::{
    layers_list, ApplicationInfo, Instance, InstanceExtensions, PhysicalDevice, Version,
};
use vulkano::swapchain::Surface;
use vulkano::VulkanObject;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family: i32,
}

impl Default for QueueFamilyIndices {
    fn default() -> Self {
        Self {
            graphics_family: -1,
            present_family: -1,
        }
    }
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0
    }
}

struct HelloTriangle {
    // sdl stuff
    sdl_ctx: sdl2::Sdl,

    // Vulkan stuff

    // Vulkan instance
    instance: Arc<Instance>,
    surface: Arc<Surface<Window>>,

    // Physical Device
    physical_device_index: usize,

    // Logical Device
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    // Debug stuff for vulkan. Hook to validation layers.
    debug_callback: Option<DebugCallback>,
}

impl Default for HelloTriangle {
    fn default() -> Self {
        // Sdl
        let sdl_ctx = Self::init_sdl_ctx();
        let instance = Self::init_instance();
        let surface = Self::init_surface(&sdl_ctx, &instance);

        // Vulkan
        let debug_callback = Self::init_debug_callback(&instance);
        let physical_device_index = Self::init_physical_device(&instance, &surface);

        let (device, graphics_queue, present_queue) =
            Self::init_logical_device(&instance, &surface, physical_device_index);

        Self {
            sdl_ctx,

            instance,
            surface,
            physical_device_index,

            device,
            graphics_queue,
            present_queue,

            debug_callback,
        }
    }
}

impl HelloTriangle {
    fn init_sdl_ctx() -> sdl2::Sdl {
        let ctx = sdl2::init().unwrap();
        ctx
    }

    fn init_surface(ctx: &sdl2::Sdl, instance: &Arc<Instance>) -> Arc<Surface<Window>> {
        let video_subsystem = ctx.video().unwrap();
        let window = video_subsystem
            .window("Vulkan", WIDTH, HEIGHT)
            .vulkan()
            .build()
            .unwrap();

        let surface_hnd = window
            .vulkan_create_surface(instance.internal_object())
            .unwrap();
        let surface =
            unsafe { Surface::from_raw_surface(instance.to_owned(), surface_hnd, window) };

        Arc::new(surface)
    }

    fn init_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            println!("Validation layers requested, but not available!")
        }

        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions");
        println!("Supported extensions: {:?}", supported_extensions);

        let app_info = ApplicationInfo {
            application_name: Some("HelloTriangle".into()),
            application_version: Some(Version {
                major: 1,
                minor: 0,
                patch: 0,
            }),
            engine_name: Some("No engine".into()),
            engine_version: Some(Version {
                major: 1,
                minor: 0,
                patch: 0,
            }),
        };

        let required_extensions = Self::get_required_extensions();

        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            let required_extensions = vulkano_win::required_extensions();
            return Instance::new(
                Some(&app_info),
                &required_extensions,
                VALIDATION_LAYERS.iter().cloned(),
            )
            .expect("failed to create Vulkan instance");
        }

        return Instance::new(Some(&app_info), &required_extensions, None)
            .expect("failed to create Vulkan instance");
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list()
            .unwrap()
            .map(|layer| layer.name().to_owned())
            .collect();
        VALIDATION_LAYERS
            .iter()
            .all(|l_name| layers.contains(&l_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();
        if ENABLE_VALIDATION_LAYERS {
            // TODO!: this should be ext_debug_utils (_report is deprecated), but that doesn't exist yet in vulkano
            extensions.ext_debug_report = true;
        }

        extensions
    }

    fn init_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("failed to find a suitable GPU!")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(device);
        let extensions_supported = Self::check_device_extension_support(device);

        let swap_chain_adequate = if extensions_supported {
            let capabilities = surface
                .capabilities(*device)
                .expect("failed to get surface validation");

            !capabilities.supported_formats.is_empty()
                && capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = device_extensions();
        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn find_queue_families(device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::default();

        for (i, queue_fam) in device.queue_families().enumerate() {
            if queue_fam.supports_graphics() {
                indices.graphics_family = i as i32;
            }
            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn init_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let indices = Self::find_queue_families(&physical_device);

        let families = [indices.graphics_family, indices.present_family];
        let unique_queue_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;
        let queue_families = unique_queue_families.iter().map(|i| {
            (
                physical_device.queue_families().nth(**i as usize).unwrap(),
                queue_priority,
            )
        });

        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            &DeviceExtensions::none(),
            queue_families,
        )
        .expect("failed to create logical device");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn init_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let msg_types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: false,
            debug: true,
        };
        DebugCallback::new(&instance, msg_types, |msg| {
            println!("validation layer: {:?}", msg.description);
        })
        .ok()
    }

    pub fn run(&mut self) {
        let mut event_pump = self.sdl_ctx.event_pump().unwrap();

        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => break 'running,
                    _ => {}
                }
            }
        }
    }
}

/// Required device extensions
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..vulkano::device::DeviceExtensions::none()
    }
}

fn main() {
    let mut app = HelloTriangle::default();

    println!("Running application!");

    app.run();
}
