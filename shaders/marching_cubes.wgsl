struct Uniforms {
    marchOrigin: vec3<f32>,
    marchDimensions: vec3<f32>,
    marchResolution: vec3<u32>,
    gradientSampleDelta: vec3<f32>,
    isolevel: f32
};

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>
};

struct Counters {
    vertexCount: atomic<u32>,
    indexCount: atomic<u32>
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volumeSampler: sampler;
@group(0) @binding(2) var volume: texture_3d<f32>;

@group(0) @binding(3) var<storage, read> edgeTable: array<u32>;
@group(0) @binding(4) var<storage, read> triTable: array<u32>;

@group(0) @binding(5) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(6) var<storage, read_write> indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: Counters;

fn approximateVertex(isolevel: f32, p0: vec3<f32>, p1: vec3<f32>, v0: f32, v1: f32) -> vec3<f32> {
    return mix(p0, p1, (isolevel - v0) / (v1 - v0));
}

fn approximateNormal(s: vec3<f32>, dsdp: vec3<f32>) -> vec3<f32> {
    let delta = vec4<f32>(uniforms.gradientSampleDelta, 0.0);
    return normalize((vec3<f32>(
        textureSampleLevel(volume, volumeSampler, s + delta.xww, 0)[0],
        textureSampleLevel(volume, volumeSampler, s + delta.wyw, 0)[0],
        textureSampleLevel(volume, volumeSampler, s + delta.wwz, 0)[0]
    ) - textureSampleLevel(volume, volumeSampler, s, 0)[0]) * dsdp);
}

@compute @workgroup_size(4, 4, 4)
fn marchingCubes(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= uniforms.marchResolution)) { return; }

    let marchOrigin = uniforms.marchOrigin;
    let marchDimensions = uniforms.marchDimensions;
    let marchResolution = uniforms.marchResolution;
    let isolevel = uniforms.isolevel;

    let sampleStep = 1.0 / vec3<f32>(marchResolution);
    let sampleStart = vec3<f32>(id) * sampleStep;
    let sampleEnd = sampleStart + sampleStep;

    let samplePoints = array(
        select(sampleStart, sampleEnd, vec3<bool>(false, false, false)),
        select(sampleStart, sampleEnd, vec3<bool>(true, false, false)),
        select(sampleStart, sampleEnd, vec3<bool>(true, false, true)),
        select(sampleStart, sampleEnd, vec3<bool>(false, false, true)),
        select(sampleStart, sampleEnd, vec3<bool>(false, true, false)),
        select(sampleStart, sampleEnd, vec3<bool>(true, true, false)),
        select(sampleStart, sampleEnd, vec3<bool>(true, true, true)),
        select(sampleStart, sampleEnd, vec3<bool>(false, true, true))
    );

    let isovalues = array(
        textureSampleLevel(volume, volumeSampler, samplePoints[0], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[1], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[2], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[3], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[4], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[5], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[6], 0)[0],
        textureSampleLevel(volume, volumeSampler, samplePoints[7], 0)[0]
    );

    var cubeIndex = 0u;
    if (isovalues[0] < isolevel) { cubeIndex |= 1u; }
    if (isovalues[1] < isolevel) { cubeIndex |= 2u; }
    if (isovalues[2] < isolevel) { cubeIndex |= 4u; }
    if (isovalues[3] < isolevel) { cubeIndex |= 8u; }
    if (isovalues[4] < isolevel) { cubeIndex |= 16u; }
    if (isovalues[5] < isolevel) { cubeIndex |= 32u; }
    if (isovalues[6] < isolevel) { cubeIndex |= 64u; }
    if (isovalues[7] < isolevel) { cubeIndex |= 128u; }

    let edgeCrossings = edgeTable[cubeIndex];
    if (edgeCrossings == 0u) { return; }

    let meshletInfoOffset = cubeIndex << 4u;
    let meshletIndicesOffset = meshletInfoOffset + 1u;
    let meshletIndexCount = triTable[meshletInfoOffset];

    let ownVerticesOffset = atomicAdd(&counters.vertexCount, countOneBits(edgeCrossings));
    let ownIndicesOffset = atomicAdd(&counters.indexCount, meshletIndexCount);

    let dsdp = 1.0 / marchDimensions;
    var vi = ownVerticesOffset;
    if ((edgeCrossings & 1u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[0], samplePoints[1], isovalues[0], isovalues[1]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 2u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[1], samplePoints[2], isovalues[1], isovalues[2]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 4u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[2], samplePoints[3], isovalues[2], isovalues[3]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 8u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[3], samplePoints[0], isovalues[3], isovalues[0]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 16u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[4], samplePoints[5], isovalues[4], isovalues[5]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 32u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[5], samplePoints[6], isovalues[5], isovalues[6]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 64u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[6], samplePoints[7], isovalues[6], isovalues[7]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 128u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[7], samplePoints[4], isovalues[7], isovalues[4]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 256u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[0], samplePoints[4], isovalues[0], isovalues[4]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 512u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[1], samplePoints[5], isovalues[1], isovalues[5]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 1024u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[2], samplePoints[6], isovalues[2], isovalues[6]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    } if ((edgeCrossings & 2048u) != 0u) {
        let samplePoint = approximateVertex(isolevel, samplePoints[3], samplePoints[7], isovalues[3], isovalues[7]);
        vertices[vi] = Vertex(fma(samplePoint, marchDimensions, marchOrigin), approximateNormal(samplePoint, dsdp));
        vi++;
    }

    for (var i = 0u; i < meshletIndexCount; i++) {
        indices[ownIndicesOffset + i] = ownVerticesOffset + triTable[meshletIndicesOffset + i];
    }
}