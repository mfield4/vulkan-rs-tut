#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use core::borrow::Borrow;
use std::{collections::HashSet, iter::FromIterator, sync::Arc, time::Instant};
use std::fs::read;

use cgmath::{
    Deg,
    Matrix3,
    Matrix4,
    Point3,
    Rad,
    Vector3,
};
use image::ImageFormat;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState},
    descriptor::{
        descriptor::DescriptorDesc,
        descriptor_set::PersistentDescriptorSet,
        DescriptorSet,
        PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Features, Queue},
    format::Format,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{Dimensions, ImageUsage, ImmutableImage, swapchain::SwapchainImage},
    instance::{
        ApplicationInfo, Instance, InstanceExtensions, layers_list, PhysicalDevice, Version,
    },
    instance::debug::{DebugCallback, MessageTypes},
    memory::DedicatedAlloc::Buffer,
    pipeline::{
        GraphicsPipeline, GraphicsPipelineAbstract, vertex::BufferlessDefinition, vertex::BufferlessVertices,
        viewport::Viewport,
    },
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain::{
        acquire_next_image, AcquireError, Capabilities, ColorSpace, CompositeAlpha, PresentMode,
        SupportedPresentModes, Surface, Swapchain,
    },
    sync::{self, GpuFuture, SharingMode},
};
use vulkano::image::AttachmentImage;
use vulkano_win::VkSurfaceBuild;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};
use vulkano::format::ClearValue;

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

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    texCoord: [f32; 2],
}

impl Vertex {
    fn new(pos: [f32; 3], color: [f32; 3], texCoord: [f32; 2]) -> Self {
        Self { pos, color, texCoord }
    }
}

impl_vertex!(Vertex, pos, color, texCoord);

#[derive(Copy, Clone)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

fn vertices() -> [Vertex; 8] {
    [
        Vertex::new([-0.5, -0.5, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0]),
        Vertex::new([0.5, -0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
        Vertex::new([0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
        Vertex::new([-0.5, 0.5, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0]),
        Vertex::new([-0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 0.0]),
        Vertex::new([0.5, -0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0]),
        Vertex::new([0.5, 0.5, -0.5], [0.0, 0.0, 1.0], [0.0, 1.0]),
        Vertex::new([-0.5, 0.5, -0.5], [1.0, 1.0, 1.0], [1.0, 1.0])
    ]
}

fn indices() -> [u16; 12] {
    [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4]
}


struct HelloTriangle {
    events_loop: EventsLoop,

    degrees: f32,

    // Vulkan stuff

    // Vulkan instance
    instance: Arc<Instance>,
    surface: Arc<Surface<Window>>,
    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    // NOTE: We need to the full type of
    // self.graphics_pipeline, because `BufferlessVertices` only
    // works when the concrete type of the graphics pipeline is visible
    // to the command buffer.
    graphics_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    swap_chain_framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<TypedBufferAccess<Content=[u16]> + Send + Sync>,
    uniform_buffers: CpuBufferPool<UniformBufferObject>,
    descriptor_set: Arc<DescriptorSet + Send + Sync>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,
    previous_frame_end: Option<Box<GpuFuture>>,
    recreate_swap_chain: bool,
    start_time: Instant,

    texture: Arc<ImmutableImage<Format>>,
    sampler: Arc<Sampler>,

    // Physical Device
    physical_device_index: usize,

    // Logical Device
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    // Debug stuff for vulkan. Hook to validation layers.
    #[allow(unused)]
    debug_callback: Option<DebugCallback>,
}

impl Default for HelloTriangle {
    fn default() -> Self {
        // First step in Vulkan is to instantiate an instance.
        let instance = Self::init_instance();

        // Next, we instantiate our window surface and events with winit.
        let (events_loop, surface) = Self::init_winit(&instance);

        // Here, we choose which physical device shall be used.
        let physical_device_index = Self::init_physical_device(&instance, &surface);

        // At this step, we create a logical device out of the physical device.
        // This also includes the queues that will be used to communicate with the device.
        let (device, graphics_queue, present_queue) =
            Self::init_logical_device(&instance, &surface, physical_device_index);

        // This method instantiates the swap chain. The swap chain hold the buffers that are ultimately sent to the screen.
        let (swap_chain, swap_chain_images) = Self::init_swap_chain(
            &instance,
            &surface,
            physical_device_index,
            &device,
            &graphics_queue,
            &present_queue,
            None,
        );

        // At this point we instantiate the render pass. The render pass describes to the graphics pipeline things like
        // output direction, color layout, and depth/stencil information.
        let render_pass = Self::init_render_pass(&device, swap_chain.format());

        // Now we finally create the graphics pipeline. Specifies the entire pipeline upfront.
        let graphics_pipeline = Self::init_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        // The framebuffers that go with the created render pass.
        let swap_chain_framebuffers = Self::init_framebuffers(&device, swap_chain.dimensions(), &swap_chain_images, &render_pass);

        let start_time = Instant::now();

        let vertex_buffer = Self::init_vertex_buffer(&graphics_queue);
        let index_buffer = Self::init_index_buffer(&graphics_queue);

        let previous_frame_end = Some(Self::init_sync_objects(&device));

        let (texture, tex_future) = {
            let image = image::load_from_memory_with_format(include_bytes!("/home/mfield/Projects/Rust/vulkan/vertex_buffer/src/test.png"), ImageFormat::PNG).unwrap().to_rgba();
            let image_data = image.into_raw().clone();


            ImmutableImage::from_iter(
                image_data.iter().cloned(),
                Dimensions::Dim2d { width: 93, height: 93 },
                Format::R8G8B8A8Srgb,
                graphics_queue.clone(),
            ).unwrap()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
                                   MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
                                   SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let (uniform_buffers, descriptor_set) = Self::init_uniform_buffers(
            device.clone(),
            graphics_pipeline.clone(),
            graphics_queue.clone(),
            texture.clone(),
            sampler.clone(),
            swap_chain_images.len(),
            start_time,
            swap_chain.dimensions(),
        );

        // Finally, we instantiate the debug callback. This function handles if we want it or not.
        let debug_callback = Self::init_debug_callback(&instance);

        let mut app = Self {
            events_loop,

            degrees: 0.0,
            instance,
            surface,
            swap_chain,
            swap_chain_images,
            render_pass,
            graphics_pipeline,
            swap_chain_framebuffers,
            vertex_buffer,
            index_buffer,
            uniform_buffers,
            descriptor_set,
            command_buffers: vec![],
            previous_frame_end,
            recreate_swap_chain: false,
            start_time,
            texture,
            sampler,
            physical_device_index,

            device,
            graphics_queue,
            present_queue,

            debug_callback,
        };

        // This is where we instantiate the command buffers.
        app.init_command_buffers();
        app
    }
}


impl HelloTriangle {
    /*Public fns*/

    pub fn run(&mut self) {
        loop {
            self.draw_frame();

            let mut done = false;
            self.events_loop.poll_events(|ev| {
                if let Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } = ev
                {
                    done = true
                }
            });
            if done {
                return;
            }
        }
    }

    pub fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }


        let (image_index, acquire_future) = match acquire_next_image(self.swap_chain.clone(), None)
            {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swap_chain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        println!("image index = {}", image_index);

        self.update_uniform_buffer(image_index);

        let command_buffer = self.command_buffers[image_index].clone();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.present_queue.clone(),
                self.swap_chain.clone(),
                image_index,
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
            Err(err) => {
                println!("{:?}", err);
                self.previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    /*
    To instantiate a Vulkan instance we do the following:
        1. Initialized application info
        2. Get required extensions, by using vulkano-win
        3. Add validation layers, if enabled.
    */
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

    /*
     To instantiate a Window we do the following:
        1. Create the events loop
        2. Create the window surface, tying it to the vulkan surface.
    */
    fn init_winit(instance: &Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>) {
        let events_loop = EventsLoop::new();
        let surface = WindowBuilder::new()
            .with_title("Vulkan")
            .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
            .build_vk_surface(&events_loop, instance.clone())
            .expect("failed to create window surface!");
        (events_loop, surface)
    }

    /*
    To instantiate and choose our physical device we have to do the following.
        1. Filter devices without request features.
        2. Filter devices unable to support the surface.
        3. Leave the rest up to the user/some priority heuristic.
    */
    fn init_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("failed to find a suitable GPU!")
    }

    fn init_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let indices = Self::find_queue_families(surface, &physical_device);

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
            &device_extensions(),
            queue_families,
        )
            .expect("failed to create logical device!");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn init_swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
        old_swap_chain: Option<Arc<Swapchain<Window>>>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let capabilities = surface
            .capabilities(physical_device)
            .expect("Failed to get surface capabilities!");

        let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.is_some()
            && image_count < capabilities.max_image_count.unwrap()
        {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let indices = Self::find_queue_families(&surface, &physical_device);

        let sharing: SharingMode = if indices.graphics_family != indices.present_family {
            vec![graphics_queue, present_queue].as_slice().into()
        } else {
            graphics_queue.into()
        };

        let (swap_chain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            image_count,
            surface_format.0,
            extent,
            1,
            image_usage,
            sharing,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            true,
            old_swap_chain.as_ref(),
        )
            .expect("failed to create swap chain!");

        (swap_chain, images)
    }

    fn init_vertex_buffer(graphics_queue: &Arc<Queue>) -> Arc<BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices().iter().cloned(), BufferUsage::all(), graphics_queue.clone()).unwrap();

        future.flush().unwrap();
        buffer
    }

    fn init_index_buffer(graphics_queue: &Arc<Queue>) -> Arc<TypedBufferAccess<Content=[u16]> + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices().iter().cloned(), BufferUsage::all(), graphics_queue.clone()).unwrap();

        future.flush().unwrap();
        buffer
    }

    fn init_uniform_buffers(
        device: Arc<Device>,
        graphics_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
        graphics_queue: Arc<Queue>,
        texture: Arc<ImmutableImage<Format>>,
        sampler: Arc<Sampler>,
        num_buffers: usize,
        start_time: Instant,
        dimensions_u32: [u32; 2],
    ) -> (CpuBufferPool<UniformBufferObject>, Arc<DescriptorSet + Send + Sync>) {
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let uniform_buffer = Self::uniform_buffer(start_time, dimensions);
        let buffer = CpuBufferPool::uniform_buffer(device.clone());
//

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(graphics_pipeline.clone(), 0)
                .add_buffer(buffer.next(uniform_buffer).unwrap()).unwrap()
                .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
                .build().unwrap()
        );
//
//        let descriptor_set_2 = Arc::new(
//            PersistentDescriptorSet::start(graphics_pipeline.clone(), 0)
//                .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
//                .build().unwrap()
//        );

        (buffer, descriptor_set)
    }

    fn init_sync_objects(device: &Arc<Device>) -> Box<GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<GpuFuture>
    }

    fn init_render_pass(
        device: &Arc<Device>,
        color_format: Format,
    ) -> Arc<RenderPassAbstract + Send + Sync> {
        Arc::new(single_pass_renderpass!(device.clone(),
                                         attachments: {
            color: {
                load: Clear,
                store: Store,
                format: color_format,
                samples: 1,
            },
            depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
         ).unwrap(), )
    }

    fn init_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "/home/mfield/Projects/Rust/vulkan/vertex_buffer/shaders/triangle.vert"
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "/home/mfield/Projects/Rust/vulkan/vertex_buffer/shaders/triangle.frag"
            }
        }

        let vert_shader_module =
            vertex_shader::Shader::load(device.clone()).expect("failed to create vertex shader");
        let frag_shader_module = fragment_shader::Shader::load(device.clone())
            .expect("failed to create fragment shader");

        let dimensions = [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vert_shader_module.main_entry_point(), ())
                .triangle_list()
                .primitive_restart(false)
                .viewports(vec![viewport])
                .fragment_shader(frag_shader_module.main_entry_point(), ())
                .depth_clamp(false)
                .polygon_mode_fill()
                .line_width(1.0) // default
                .cull_mode_back()
                .depth_stencil_simple_depth()
                .front_face_counter_clockwise()
                .blend_pass_through()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        )
    }

    fn init_framebuffers(
        device: &Arc<Device>,
        dimensions: [u32; 2],
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

        swap_chain_images
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(image.clone()).unwrap()
                        .add(depth_buffer.clone()).unwrap()
                        .build().unwrap(),
                ) as Arc<FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>()
    }

    fn init_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();


        self.command_buffers = self
            .swap_chain_framebuffers
            .iter()
            .map(|framebuffer| {
                Arc::new(
                    AutoCommandBufferBuilder::primary_simultaneous_use(
                        self.device.clone(),
                        queue_family,
                    )
                        .unwrap()
                        .begin_render_pass(
                            framebuffer.clone(),
                            false,
                            vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()],
                        )
                        .unwrap()
                        .draw_indexed(self.graphics_pipeline.clone(), &DynamicState::none(), vec![self.vertex_buffer.clone()], self.index_buffer.clone(), (self.descriptor_set.clone()), ())
                        .unwrap()
                        .end_render_pass()
                        .unwrap()
                        .build()
                        .unwrap(),
                )
            })
            .collect()
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

    /* private fns */

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list()
            .unwrap()
            .map(|layer| layer.name().to_owned())
            .collect();
        VALIDATION_LAYERS
            .iter()
            .all(|l_name| layers.contains(&l_name.to_string()))
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = device_extensions();
        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn choose_swap_surface_format(
        available_formats: &[(Format, ColorSpace)],
    ) -> (Format, ColorSpace) {
        *available_formats
            .iter()
            .find(|(format, color_space)| {
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            })
            .unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            current_extent
        } else {
            let mut actual_extent = [WIDTH, HEIGHT];
            actual_extent[0] = capabilities.min_image_extent[0]
                .max(capabilities.max_image_extent[0].min(actual_extent[0]));
            actual_extent[1] = capabilities.min_image_extent[1]
                .max(capabilities.max_image_extent[1].min(actual_extent[1]));

            actual_extent
        }
    }

    fn find_queue_families(
        surface: &Arc<Surface<Window>>,
        device: &PhysicalDevice,
    ) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::default();

        for (i, queue_fam) in device.queue_families().enumerate() {
            if queue_fam.supports_graphics() {
                indices.graphics_family = i as i32;
            }

            if surface.is_supported(queue_fam).unwrap() {
                indices.present_family = i as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();
        if ENABLE_VALIDATION_LAYERS {
            // TODO!: this should be ext_debug_utils (_report is deprecated), but that doesn't exist yet in vulkano
            extensions.ext_debug_report = true;
        }

        extensions
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
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

    fn recreate_swap_chain(&mut self) {
        let (swap_chain, images) = Self::init_swap_chain(
            &self.instance,
            &self.surface,
            self.physical_device_index,
            &self.device,
            &self.graphics_queue,
            &self.present_queue,
            Some(self.swap_chain.clone()),
        );
        self.swap_chain = swap_chain;
        self.swap_chain_images = images;

        self.render_pass = Self::init_render_pass(&self.device, self.swap_chain.format());
        self.graphics_pipeline = Self::init_graphics_pipeline(
            &self.device,
            self.swap_chain.dimensions(),
            &self.render_pass,
        );
        self.swap_chain_framebuffers =
            Self::init_framebuffers(&self.device, self.swap_chain.dimensions(), &self.swap_chain_images, &self.render_pass);

        let (uni_buffers, descriptor_set) = Self::init_uniform_buffers(
            self.device.clone(),
            self.graphics_pipeline.clone(),
            self.graphics_queue.clone(),
            self.texture.clone(),
            self.sampler.clone(),
            self.swap_chain_images.len(),
            Instant::now(), self.swap_chain.dimensions(),
        );

        self.uniform_buffers = uni_buffers;
        self.descriptor_set = descriptor_set;
        self.init_command_buffers();
    }

    fn uniform_buffer(start_time: Instant, dimensions: [f32; 2]) -> UniformBufferObject {
        let elapsed = Instant::now().duration_since(start_time);
        let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;

        println!("Rotation: {}", rotation);

        let rotation = Matrix3::from_angle_z(Rad(rotation as f32));


        // note: this teapot was meant for OpenGL where the origin is at the lower left
        //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
        let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;

        let mut proj = cgmath::perspective(Deg(45.0), aspect_ratio, 0.01, 10.0);
        proj.y.y *= -1.0;

        let view = Matrix4::look_at(Point3::new(2.0, 2.0, 2.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));

        let scale = Matrix4::from_scale(0.5);

        UniformBufferObject {
            model: Matrix4::from(rotation).into(),
            view: (view * scale).into(),
            proj: proj.into(),
        }
    }

    fn update_uniform_buffer(&mut self, image_index: usize) {
        let dimensions_u32 = self.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let uniform_buffer = Self::uniform_buffer(self.start_time, dimensions);

        self.descriptor_set = Arc::new(
            PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 0)
                .add_buffer(self.uniform_buffers.next(uniform_buffer).unwrap()).unwrap()
                .add_sampled_image(self.texture.clone(), self.sampler.clone()).unwrap()
                .build().unwrap()
        );
//
//        self.descriptor_set_2 = Arc::new(PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 0)
//            .build().unwrap()
//        );

        self.init_command_buffers();
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
