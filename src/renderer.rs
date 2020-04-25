use gfx_hal::{
    adapter, buffer as b, command, format as f, image as i, pass, pool,
    prelude::*,
    pso,
    queue::{family::QueueFamilyId, Submission},
    window, Backend,
};

use std::borrow::Borrow;
use std::iter;
use std::mem::ManuallyDrop;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

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
    command_buffers: Option<Vec<B::CommandBuffer>>,
    submission_complete_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    command_pool: ManuallyDrop<B::CommandPool>,
    descriptor_set: Option<ManuallyDrop<DescriptorSet<'a, B>>>,
    memory: ManuallyDrop<Memory<'a, B, Vertex>>,
    uniform_memory: ManuallyDrop<Memory<'a, B, f32>>,
    swapchain: ManuallyDrop<Swapchain<'a, B>>,
    render_pass: ManuallyDrop<B::RenderPass>,
    pipeline: ManuallyDrop<Pipeline<'a, B>>,
    resized: Arc<AtomicBool>,
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
        resized: Arc<AtomicBool>,
    ) -> Self {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        let descriptor_set = DescriptorSet::new(device);
        let vertex_buffer = Buffer::new(device, TRIANGLE.to_vec(), b::Usage::VERTEX, &limits);
        let memory = Memory::new(vertex_buffer, &memory_types);
        let uniform_buffer = Buffer::new(device, vec![0.0, 0.0], b::Usage::UNIFORM, &limits);
        let uniform_memory = Memory::new(uniform_buffer, &memory_types);

        unsafe {
            device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
                set: &descriptor_set.set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &*uniform_memory.buffer.buf,
                    b::SubRange::WHOLE,
                )),
            }));
        }

        let swapchain = Swapchain::new(device, surface, adapter, init_dims);
        let render_pass = Self::create_render_pass(device, swapchain.format);
        let pipeline = Pipeline::new::<Vertex>(
            device,
            "src/data/triangle.vert.spv",
            "src/data/triangle.frag.spv",
            &*render_pass,
            Some(&*descriptor_set.set_layout),
        );

        let mut command_pool = Self::create_command_pool(&device, family);
        let frames_in_flight: usize = 1;

        let command_buffers = Self::allocate_command_buffer(&mut command_pool, 1);
        let submission_complete_semaphores = Self::create_semaphores(&device, frames_in_flight);
        let submission_complete_fences = Self::create_fences(&device, frames_in_flight);

        Renderer {
            device,
            submission_complete_semaphores,
            submission_complete_fences,
            frames_in_flight,
            command_pool: ManuallyDrop::new(command_pool),
            descriptor_set: Some(ManuallyDrop::new(descriptor_set)),
            memory: ManuallyDrop::new(memory),
            uniform_memory: ManuallyDrop::new(uniform_memory),
            swapchain: ManuallyDrop::new(swapchain),
            render_pass,
            pipeline: ManuallyDrop::new(pipeline),
            command_buffers: Some(command_buffers),
            frame: 0,
            resized,
        }
    }

    pub fn render(&mut self, queue: &mut B::CommandQueue) {
        if self.resized.load(Ordering::Relaxed) {
            self.swapchain.recreate();
            self.resized.store(false, Ordering::Relaxed);
            return;
        }
        let surface_image = unsafe {
            match self.swapchain.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.swapchain.recreate();
                    return;
                }
            }
        };

        println!("{:?}", self.swapchain.dims);
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
            // if frame_idx == 0 {
            self.command_pool.reset(false);
            // }
        }

        self.uniform_memory.buffer.content[0] =
            0.1 * (self.frame as f32 * std::f32::consts::PI / 60.0).cos();
        self.uniform_memory.buffer.content[1] =
            0.1 * (self.frame as f32 * std::f32::consts::PI / 60.0).sin();
        self.uniform_memory.update_data(0);

        let cmd_buffer = &mut self.command_buffers.as_mut().unwrap()[0];
        unsafe {
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            cmd_buffer.set_viewports(0, &[self.swapchain.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.swapchain.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&self.pipeline.pipeline);
            cmd_buffer.bind_vertex_buffers(
                0,
                iter::once((&*self.memory.buffer.buf, b::SubRange::WHOLE)),
            );

            assert!(self.descriptor_set.is_some());
            if let Some(descriptor_set) = &self.descriptor_set {
                cmd_buffer.bind_graphics_descriptor_sets(
                    &self.pipeline.pipeline_layout,
                    0,
                    Some(&descriptor_set.set),
                    &[],
                );
            }
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
                command_buffers: iter::once(&cmd_buffer),
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

            self.device.destroy_framebuffer(frame_buffer);

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
            if let Some(mut descriptor_set) = self.descriptor_set.take() {
                ManuallyDrop::drop(&mut descriptor_set);
            }
            ManuallyDrop::drop(&mut self.memory);
            ManuallyDrop::drop(&mut self.uniform_memory);

            for s in self.submission_complete_semaphores.drain(..) {
                device.destroy_semaphore(s);
            }

            for f in self.submission_complete_fences.drain(..) {
                device.wait_for_fence(&f, !0).unwrap();
                device.destroy_fence(f);
            }

            if let Some(cbs) = self.command_buffers.take() {
                self.command_pool.free(cbs);
            }
            device.destroy_command_pool(ManuallyDrop::into_inner(ptr::read(&self.command_pool)));

            device.destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.pipeline);
        }
    }
}
