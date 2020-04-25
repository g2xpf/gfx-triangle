use gfx_hal::{adapter::MemoryType, memory as m, prelude::*, Backend, MemoryTypeId};
use std::iter;
use std::mem::ManuallyDrop;
use std::ptr;

use super::buffer::Buffer;

pub struct Memory<'a, B: Backend, T> {
    pub buffer: ManuallyDrop<Buffer<'a, B, T>>,
    memory: ManuallyDrop<B::Memory>,
    size: u64,
}

impl<'a, B: Backend, T> Memory<'a, B, T> {
    pub fn new(mut buffer: Buffer<'a, B, T>, memory_types: &[MemoryType]) -> Self {
        let (memory, size) = Self::allocate_gpu_memory(&mut buffer, memory_types);
        Memory {
            buffer: ManuallyDrop::new(buffer),
            memory,
            size,
        }
    }

    pub fn allocate_gpu_memory(
        buffer: &mut Buffer<'a, B, T>,
        memory_types: &[MemoryType],
    ) -> (ManuallyDrop<B::Memory>, u64) {
        let device = &buffer.device;
        unsafe {
            let buffer_req = device.get_buffer_requirements(&buffer.buf);
            let upload_type = Self::upload_type(memory_types, &buffer_req);
            let memory = device
                .allocate_memory(upload_type, buffer_req.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut buffer.buf)
                .unwrap();
            let mapping = device.map_memory(&memory, m::Segment::ALL).unwrap();
            ptr::copy_nonoverlapping(
                buffer.content.as_ptr() as *const u8,
                mapping,
                buffer.len as usize,
            );
            device
                .flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&memory);
            (ManuallyDrop::new(memory), buffer_req.size)
        }
    }

    fn upload_type(properties: &[MemoryType], buffer_req: &m::Requirements) -> MemoryTypeId {
        properties
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type
                        .properties
                        .contains(m::Properties::CPU_VISIBLE | m::Properties::COHERENT)
            })
            .unwrap()
            .into()
    }

    pub fn update_data(&mut self, offset: u64)
    where
        T: Copy,
    {
        let device = &self.buffer.device;

        let upload_size = self.buffer.memory_size();

        assert!(offset + upload_size as u64 <= self.size);
        let memory = &self.memory;

        unsafe {
            let mapping = device
                .map_memory(memory, m::Segment { offset, size: None })
                .unwrap();
            ptr::copy_nonoverlapping(
                self.buffer.content.as_ptr() as *const u8,
                mapping,
                upload_size as usize,
            );
            device.unmap_memory(memory);
        }
    }
}

impl<'a, B: Backend, T> Drop for Memory<'a, B, T> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.buffer);
            self.buffer
                .device
                .free_memory(ManuallyDrop::into_inner(ptr::read(&self.memory)))
        }
    }
}
