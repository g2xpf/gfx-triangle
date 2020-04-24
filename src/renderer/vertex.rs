const R: [f32; 3] = [1.0, 0.0, 0.0];
const G: [f32; 3] = [0.0, 1.0, 0.0];
const B: [f32; 3] = [0.0, 0.0, 1.0];

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct Vertex {
    a_Pos: [f32; 2],
    a_Color: [f32; 3],
}

pub const TRIANGLE: [Vertex; 3] = [
    Vertex {
        a_Pos: [0.0, 0.4],
        a_Color: R,
    },
    Vertex {
        a_Pos: [-0.4, -0.3],
        a_Color: G,
    },
    Vertex {
        a_Pos: [0.4, -0.3],
        a_Color: B,
    },
];
