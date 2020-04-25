use gfx_hal::{adapter::Adapter, format as f, prelude::*, pso, window, Backend};

pub struct Swapchain<'a, B: Backend> {
    device: &'a B::Device,
    adapter: &'a Adapter<B>,
    pub viewport: pso::Viewport,
    pub dims: window::Extent2D,
    pub surface: &'a mut B::Surface,
    pub format: f::Format,
}

impl<'a, B: Backend> Swapchain<'a, B> {
    pub fn new(
        device: &'a B::Device,
        surface: &'a mut B::Surface,
        adapter: &'a Adapter<B>,
        dims: window::Extent2D,
    ) -> Self {
        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == f::ChannelType::Srgb)
                .copied()
                .unwrap_or(formats[0])
        });
        let mut swap_config = window::SwapchainConfig::from_caps(&caps, format, dims);
        swap_config.present_mode = window::PresentMode::FIFO;
        let extent = swap_config.extent.to_extent();

        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        let mut swapchain = Swapchain {
            device,
            surface,
            adapter,
            viewport,
            format,
            dims,
        };

        swapchain.recreate();
        swapchain
    }

    pub fn recreate(&mut self) {
        let caps = self.surface.capabilities(&self.adapter.physical_device);

        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dims);
        let extent = swap_config.extent;
        unsafe {
            self.surface
                .configure_swapchain(&self.device, swap_config)
                .expect("Can't create swapchain");
        }

        self.viewport.rect.w = extent.width as _;
        self.viewport.rect.h = extent.height as _;
    }
}

impl<'a, B: Backend> Drop for Swapchain<'a, B> {
    fn drop(&mut self) {
        unsafe { self.surface.unconfigure_swapchain(&self.device) }
    }
}
