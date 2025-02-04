@group(0) @binding(0) var inputVelocity: texture_storage_3d<r32float, read>;
@group(1) @binding(0) var outputVelocity: texture_storage_3d<r32float, write>;
@group(2) @binding(0) var advectedDistance: texture_storage_3d<r32float, write>;
@group(3) @binding(0) var boundaryField: texture_storage_3d<r32float, read>;

@compute @workgroup_size(4, 4, 4)
fn updateVelocity(@builtin(global_invocation_id) id: vec3<u32>) {
    // Update velocity (gravity, etc.)
}

@compute @workgroup_size(4, 4, 4)
fn solveIncompressibility(@builtin(global_invocation_id) id: vec3<u32>) {
    // Solve incompressibility
}

@compute @workgroup_size(4, 4, 4)
fn advect() {
    // Advect velocity and distance fields
}