use gfx_hal::{prelude::*, pso, Backend};
use std::mem::ManuallyDrop;
use std::ptr;

pub struct DescriptorSet<'a, B: Backend> {
    device: &'a B::Device,
    pub set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    pool: ManuallyDrop<B::DescriptorPool>,
    pub set: B::DescriptorSet,
}

impl<'a, B: Backend> DescriptorSet<'a, B> {
    pub fn new(device: &'a B::Device) -> Self {
        let set_layout = Self::create_descriptor_set_layout(device);
        let mut pool = Self::create_descriptor_pool(device);
        let set = Self::create_descriptor_set(&mut pool, &set_layout);

        DescriptorSet {
            set_layout,
            pool,
            set,
            device,
        }
    }

    pub(super) fn create_descriptor_set_layout(
        device: &B::Device,
    ) -> ManuallyDrop<B::DescriptorSetLayout> {
        ManuallyDrop::new(
            unsafe {
                device.create_descriptor_set_layout(
                    &[pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: pso::DescriptorType::Buffer {
                            ty: pso::BufferDescriptorType::Uniform,
                            format: pso::BufferDescriptorFormat::Structured {
                                dynamic_offset: false,
                            },
                        },
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::VERTEX,
                        immutable_samplers: false,
                    }],
                    &[],
                )
            }
            .expect("Can't create descriptor set layout"),
        )
    }

    pub(super) fn create_descriptor_pool(device: &B::Device) -> ManuallyDrop<B::DescriptorPool> {
        ManuallyDrop::new(unsafe {
            device
                .create_descriptor_pool(
                    1,
                    &[pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Buffer {
                            ty: pso::BufferDescriptorType::Uniform,
                            format: pso::BufferDescriptorFormat::Structured {
                                dynamic_offset: false,
                            },
                        },
                        count: 1,
                    }],
                    pso::DescriptorPoolCreateFlags::empty(),
                )
                .expect("Can't create descriptor pool")
        })
    }

    pub(super) fn create_descriptor_set(
        desc_pool: &mut ManuallyDrop<B::DescriptorPool>,
        layout: &ManuallyDrop<B::DescriptorSetLayout>,
    ) -> B::DescriptorSet {
        unsafe { desc_pool.allocate_set(&layout).unwrap() }
    }
}

impl<'a, B: Backend> Drop for DescriptorSet<'a, B> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.set_layout,
                )));
            self.device
                .destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(&self.pool)));
        }
    }
}
