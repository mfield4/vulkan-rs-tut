#version 450
#extension GL_ARB_separate_shader_objects : enable


vec2 positions[3] = vec2[](
vec2(0.0, -0.5),
vec2(0.5, 0.5),
vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
vec3(1.0, 0.0, 0.0),
vec3(0.0, 1.0, 0.0),
vec3(0.0, 0.0, 1.0)
);

layout(location = 0) out vec3 fragColor;

void main() {
    // Buitin gl_position funcstions as the output.
    // Builtin gl_VertexIndex points to vertex. Usually used for vertex buffer.
    gl_Position = vec4(positions[gl_VertexIndex], 0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
