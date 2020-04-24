use gfx_hal::{
    adapter, buffer as b, command, format as f, image as i, pass, pool,
    prelude::*,
    queue::{family::QueueFamilyId, Submission},
    window, Backend,
};

use std::borrow::Borrow;
use std::iter;
use std::mem::ManuallyDrop;

mod buffer;
mod descriptor_set;
mod memory;
mod pipeline;
mod swapchain;
mod vertex;

use buffer::Buffer;
use descriptor_set::DescriptorSet;
use memory::Memory;
use pipeline::Pipeline;
use std::ptr;
use swapchain::Swapchain;
use vertex::{Vertex, TRIANGLE};

pub struct Renderer<'a, B: Backend> {
    frame: usize,
    device: &'a B::Device,
    frames_in_flight: usize,
    command_buffers: Vec<B::CommandBuffer>,
    submission_complete_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    command_pool: ManuallyDrop<B::CommandPool>,
    descriptor_set: ManuallyDrop<DescriptorSet<'a, B>>,
    memory: ManuallyDrop<Memory<'a, B, Vertex>>,
    swapchain: ManuallyDrop<Swapchain<'a, B>>,
    render_pass: ManuallyDrop<B::RenderPass>,
    pipeline: ManuallyDrop<Pipeline<'a, B>>,
}

impl<'a, B> Renderer<'a, B>
where
    B: Backend,
{
    pub fn new(
        surface: &'a mut B::Surface,
        adapter: &'a adapter::Adapter<B>,
        device: &'a B::Device,
        family: QueueFamilyId,
        init_dims: window::Extent2D,
    ) -> Self {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        let descriptor_set = DescriptorSet::new(device);
        let vertex_buffer = Buffer::new(device, &TRIANGLE, &limits);
        let memory = Memory::new(vertex_buffer, &memory_types);
        let swapchain = Swapchain::new(device, surface, adapter, init_dims);
        let render_pass = Self::create_render_pass(device, swapchain.format);
        let pipeline = Pipeline::new::<Vertex>(
            device,
            "src/data/triangle.vert.spv",
            "src/data/triangle.frag.spv",
            &*render_pass,
            &*descriptor_set.set_layout,
        );

        let mut command_pool = Self::create_command_pool(&device, family);
        let frames_in_flight: usize = 2;

        let command_buffers = Self::allocate_command_buffer(&mut command_pool, frames_in_flight);
        let submission_complete_semaphores = Self::create_semaphores(&device, frames_in_flight);
        let submission_complete_fences = Self::create_fences(&device, frames_in_flight);

        Renderer {
            device,
            submission_complete_semaphores,
            submission_complete_fences,
            frames_in_flight,
            command_pool: ManuallyDrop::new(command_pool),
            descriptor_set: ManuallyDrop::new(descriptor_set),
            memory: ManuallyDrop::new(memory),
            swapchain: ManuallyDrop::new(swapchain),
            render_pass,
            pipeline: ManuallyDrop::new(pipeline),
            command_buffers,
            frame: 0,
        }
    }

    pub fn render(&mut self, queue: &mut B::CommandQueue) {
        let surface_image = unsafe {
            match self.swapchain.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.swapchain.recreate();
                    return;
                }
            }
        };

        let frame_buffer = unsafe {
            self.device.create_framebuffer(
                &self.render_pass,
                iter::once(surface_image.borrow()),
                i::Extent {
                    width: self.swapchain.dims.width,
                    height: self.swapchain.dims.height,
                    depth: 1,
                },
            )
        }
        .expect("Could not create frame buffer");

        let frame_idx = self.frame % self.frames_in_flight;

        unsafe {
            let fence = &self.submission_complete_fences[frame_idx];
            self.device
                .wait_for_fence(fence, !0)
                .expect("Can't wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Can't wait for fence");
            if frame_idx == 0 {
                self.command_pool.reset(false);
            }
        }

        let cmd_buffer = &mut self.command_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            cmd_buffer.set_viewports(0, &[self.swapchain.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.swapchain.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&self.pipeline.pipeline);
            cmd_buffer.bind_vertex_buffers(
                0,
                iter::once((&*self.memory.buffer.buf, b::SubRange::WHOLE)),
            );
            cmd_buffer.bind_graphics_descriptor_sets(
                &self.pipeline.pipeline_layout,
                0,
                iter::once(&self.descriptor_set.set),
                &[],
            );
            cmd_buffer.begin_render_pass(
                &self.render_pass,
                &frame_buffer,
                self.swapchain.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [0.8, 0.8, 0.8, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );
            cmd_buffer.draw(0..3, 0..1);
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: iter::once(&*cmd_buffer),
                wait_semaphores: None,
                signal_semaphores: iter::once(&self.submission_complete_semaphores[frame_idx]),
            };

            queue.submit(
                submission,
                Some(&self.submission_complete_fences[frame_idx]),
            );

            let result = queue.present_surface(
                &mut self.swapchain.surface,
                surface_image,
                Some(&self.submission_complete_semaphores[frame_idx]),
            );

            if result.is_err() {
                self.swapchain.recreate();
            }
        }

        self.frame += 1;
    }

    fn create_render_pass(device: &B::Device, format: f::Format) -> ManuallyDrop<B::RenderPass> {
        let attachment = pass::Attachment {
            format: Some(format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: i::Layout::Undefined..i::Layout::Present,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, i::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        ManuallyDrop::new(
            unsafe { device.create_render_pass(&[attachment], &[subpass], &[]) }
                .expect("Can't create render pass"),
        )
    }

    fn create_command_pool(device: &B::Device, family: QueueFamilyId) -> B::CommandPool {
        unsafe { device.create_command_pool(family, pool::CommandPoolCreateFlags::empty()) }
            .expect("Can't create command pooll")
    }

    fn allocate_command_buffer(
        command_pool: &mut B::CommandPool,
        frames_in_flight: usize,
    ) -> Vec<B::CommandBuffer> {
        let mut v = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            v.push(unsafe { command_pool.allocate_one(command::Level::Primary) });
        }
        v
    }

    fn create_semaphores(device: &B::Device, frames_in_flight: usize) -> Vec<B::Semaphore> {
        let mut v = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            v.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
        }
        v
    }

    fn create_fences(device: &B::Device, frames_in_flight: usize) -> Vec<B::Fence> {
        let mut v = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            v.push(device.create_fence(true).expect("Could not create fence"));
        }
        v
    }
}

impl<'a, B: Backend> Drop for Renderer<'a, B> {
    fn drop(&mut self) {
        let device = &self.device;
        device.wait_idle().unwrap();
        unsafe {
            ManuallyDrop::drop(&mut self.descriptor_set);
            ManuallyDrop::drop(&mut self.memory);
            device.destroy_command_pool(ManuallyDrop::into_inner(ptr::read(&self.command_pool)));
            for s in self.submission_complete_semaphores.drain(..) {
                device.destroy_semaphore(s);
            }

            for f in self.submission_complete_fences.drain(..) {
                device.destroy_fence(f);
            }

            ManuallyDrop::drop(&mut self.render_pass);
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.pipeline);
        }
    }
}
