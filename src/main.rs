#[cfg(feature = "metal")]
use gfx_backend_metal as back;

#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

mod renderer;
use renderer::Renderer;

use std::sync::atomic::{AtomicBool, Ordering};

use gfx_hal::{prelude::*, window, Features};
use std::thread;

use std::sync::Arc;

const DIMS: window::Extent2D = window::Extent2D {
    width: 1024,
    height: 768,
};

fn main() {
    env_logger::init();
    #[cfg(debug_assersion)]
    let mut fps_counter = fps_counter::FPSCounter::new();
    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_title("triangle")
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )));
    let window = wb.build(&event_loop).expect("failed to create window");

    let window_should_closed = Arc::new(AtomicBool::new(false));
    let resized = Arc::new(AtomicBool::new(false));

    let window_should_closed_mutex = Arc::clone(&window_should_closed);
    let resized_cloned = Arc::clone(&resized);
    let handler = thread::spawn(move || {
        let instance = back::Instance::create("gfx-rs triangle", 1)
            .expect("failed to create an instance of gfx");
        let mut adapters = instance.enumerate_adapters();
        let mut surface = unsafe {
            instance
                .create_surface(&window)
                .expect("failed to create a surface")
        };
        let adapter = adapters.remove(0);

        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();
        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(&family, &[1.0])], Features::empty())
                .unwrap()
        };

        let mut queue_group = gpu.queue_groups.pop().unwrap();
        let queue = &mut queue_group.queues[0];
        let device = gpu.device;

        {
            let mut renderer = Renderer::new(
                &mut surface,
                &adapter,
                &device,
                queue_group.family,
                DIMS,
                resized_cloned,
            );

            while !window_should_closed_mutex.load(Ordering::Relaxed) {
                #[cfg(debug_assersion)]
                println!("frame: {}", fps_counter.tick());
                renderer.render(queue);
            }
        }

        unsafe {
            instance.destroy_surface(surface);
        }
    });

    let mut handler = Some(handler);

    event_loop.run(move |event, _, control_flow| {
        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                winit::event::WindowEvent::CloseRequested
                | winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => {
                    window_should_closed.store(true, Ordering::Relaxed);
                    println!("closed");
                    if let Some(handler) = handler.take() {
                        handler.join().unwrap();
                    }
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                winit::event::WindowEvent::Resized(_) => {
                    resized.store(true, Ordering::Relaxed);
                }
                _ => {}
            }
        }
    });
}
