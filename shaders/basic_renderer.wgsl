struct Light {
    position: vec3<f32>,
    color: vec3<f32>
};

struct Material {
    ambientColor: vec3<f32>,
    diffuseColor: vec3<f32>,
    specularColor: vec3<f32>,
    shininess: f32
};

struct Uniforms {
    modelMatrix: mat4x4<f32>,
    normalMatrix: mat4x4<f32>,
    cameraMatrix: mat4x4<f32>,
    cameraPosition: vec3<f32>
};

struct VSInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>
};

struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldSpacePosition: vec3<f32>,
    @location(1) worldSpaceNormal: vec3<f32>
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex fn processVertex(vsInput: VSInput) -> VSOutput {
    var vsOutput: VSOutput;
    var worldSpacePosition = uniforms.modelMatrix * vec4<f32>(vsInput.position, 1.0);
    vsOutput.worldSpacePosition = worldSpacePosition.xyz;
    vsOutput.worldSpaceNormal = (uniforms.normalMatrix * vec4<f32>(vsInput.normal, 0.0)).xyz;
    vsOutput.position = uniforms.cameraMatrix * worldSpacePosition;
    return vsOutput;
}

@fragment fn processFragment(fsInput: VSOutput) -> @location(0) vec4<f32> {
    var fragPosition = fsInput.worldSpacePosition;
    var fragNormal = normalize(fsInput.worldSpaceNormal);
    var viewDir = normalize(uniforms.cameraPosition - fragPosition);

    var light: Light;
    light.position = vec3<f32>(2.0, 2.0, 2.0);
    light.color = vec3<f32>(2.0);

    var material: Material;
    material.diffuseColor = 0.5 + 0.5 * fragNormal;
    material.ambientColor = 0.05 * material.diffuseColor;
    material.specularColor = vec3<f32>(1.0);
    material.shininess = 32.0;

    var lightRelPos = light.position - fragPosition;
    var lightDir = normalize(lightRelPos);
    var reflectDir = -reflect(lightDir, fragNormal);
    var illum = light.color / dot(lightRelPos, lightRelPos);
    var diffuse = max(0.0, dot(lightDir, fragNormal));
    var specular = select(pow(max(0.0, dot(reflectDir, viewDir)), material.shininess), 0.0, diffuse == 0.0);
    var shade = (material.ambientColor + diffuse * material.diffuseColor + specular * material.specularColor) * illum;

    return vec4<f32>(pow(shade, vec3<f32>(1.0 / 2.2)), 1.0);
}