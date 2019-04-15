extern crate sdl2;
extern crate vulkano;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

struct HelloTriangle {
    sdl_ctx: sdl2::Sdl,
    window: sdl2::video::Window,
}

impl Default for HelloTriangle {
    fn default() -> HelloTriangle {
        let sdl_ctx = HelloTriangle::init_sdl_ctx();
        let window = HelloTriangle::init_window(&sdl_ctx);

        HelloTriangle { sdl_ctx, window }
    }
}

impl HelloTriangle {
    pub fn init_sdl_ctx() -> sdl2::Sdl {
        let ctx = sdl2::init().unwrap();
        ctx
    }

    pub fn init_window(ctx: &sdl2::Sdl) -> sdl2::video::Window {
        let video_subsystem = ctx.video().unwrap();
        let window = video_subsystem
            .window("Vulkan", WIDTH, HEIGHT)
            .vulkan()
            .build()
            .unwrap();

        window
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

fn main() {
    let mut app = HelloTriangle::default();

    println!("Running application!");

    app.run();
}
