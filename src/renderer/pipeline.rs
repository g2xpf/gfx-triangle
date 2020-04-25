use gfx_hal::{format as f, pass::Subpass, prelude::*, pso, Backend};
use std::fs::read;
use std::io::Cursor;
use std::mem::{self, ManuallyDrop};
use std::ptr;

const ENTRY_NAME: &str = "main";

pub struct Pipeline<'a, B: Backend> {
    device: &'a B::Device,
    pub pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<B::PipelineLayout>,
}

impl<'a, B: Backend> Pipeline<'a, B> {
    pub fn new<T>(
        device: &'a B::Device,
        vs_path: &str,
        fs_path: &str,
        render_pass: &B::RenderPass,
        set_layout: Option<&B::DescriptorSetLayout>,
    ) -> Self {
        let pipeline_layout = ManuallyDrop::new(
            unsafe { device.create_pipeline_layout(set_layout, &[]) }
                .expect("Can't create pipeline layout"),
        );

        let vs_module = Self::load_spirv(device, vs_path);
        let fs_module = Self::load_spirv(device, fs_path);

        let (vs_entry, fs_entry) = (
            pso::EntryPoint {
                entry: ENTRY_NAME,
                module: &vs_module,
                specialization: gfx_hal::spec_const_list![0 => 0.8f32],
            },
            pso::EntryPoint {
                entry: ENTRY_NAME,
                module: &fs_module,
                specialization: pso::Specialization::default(),
            },
        );

        let shader_entries = pso::GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let subpass = Subpass {
            index: 0,
            main_pass: &*render_pass,
        };

        let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
            shader_entries,
            pso::Primitive::TriangleList,
            pso::Rasterizer::FILL,
            &*pipeline_layout,
            subpass,
        );
        pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
            mask: pso::ColorMask::ALL,
            blend: Some(pso::BlendState::ALPHA),
        });

        pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
            binding: 0,
            stride: mem::size_of::<T>() as u32,
            rate: pso::VertexInputRate::Vertex,
        });

        pipeline_desc.attributes.push(pso::AttributeDesc {
            location: 0,
            binding: 0,
            element: pso::Element {
                format: f::Format::Rg32Sfloat,
                offset: 0,
            },
        });
        pipeline_desc.attributes.push(pso::AttributeDesc {
            location: 1,
            binding: 0,
            element: pso::Element {
                format: f::Format::Rgb32Sfloat,
                offset: 8,
            },
        });

        let graphic_pipeline =
            unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }.unwrap();

        unsafe {
            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);
        }

        Pipeline {
            device,
            pipeline: ManuallyDrop::new(graphic_pipeline),
            pipeline_layout,
        }
    }

    fn load_spirv(device: &B::Device, path: &str) -> B::ShaderModule {
        let data = read(path).unwrap();
        let spirv = pso::read_spirv(Cursor::new(&data[..])).unwrap();
        unsafe { device.create_shader_module(&spirv) }.unwrap()
    }
}

impl<'a, B: Backend> Drop for Pipeline<'a, B> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_layout,
                )));
        }
    }
}
