#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 target0;

void main() {
    target0 = vec4(v_color, 1.0);
}
