use gfx_hal::{buffer, prelude::*, Backend, Limits};
use std::mem::{self, ManuallyDrop};
use std::ptr;

pub struct Buffer<'a, B: Backend, T> {
    pub device: &'a B::Device,
    pub buf: ManuallyDrop<B::Buffer>,
    pub content: &'a [T],
    pub len: u64,
}

impl<'a, B: Backend, T> Buffer<'a, B, T> {
    pub fn new(device: &'a B::Device, content: &'a [T], limits: &Limits) -> Self {
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let buffer_stride = mem::size_of::<T>() as u64;
        let buffer_len = content.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let memory_size = ((buffer_len + non_coherent_alignment - 1) / non_coherent_alignment)
            * non_coherent_alignment;

        Buffer {
            device,
            buf: ManuallyDrop::new({
                unsafe {
                    device
                        .create_buffer(memory_size, buffer::Usage::VERTEX)
                        .unwrap()
                }
            }),
            content,
            len: buffer_len,
        }
    }
}

impl<'a, B: Backend, T> Drop for Buffer<'a, B, T> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.buf)))
        }
    }
}
