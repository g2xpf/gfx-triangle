#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(constant_id = 0) const float scale = 1.0f;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;
layout(location = 0) out vec3 v_color;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    v_color = a_color;
    gl_Position = vec4(scale * a_pos, 0.0, 1.0);
}
