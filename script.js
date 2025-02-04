import Matrix4 from "./modules/matrix4.mjs";
import Stats from "./modules/stats.min.mjs";

const CAMERA_FOCAL_LENGTH = 1.5;
const CAMERA_NEAR_PLANE = 0.1;
const CAMERA_FAR_PLANE = 50.0;
const CAMERA_MOVEMENT_RATE = 4.0;
const CAMERA_YAW_PITCH_RATE = 0.004;
const CAMERA_ROLL_RATE = 3.0;
const CAMERA_CONTROL_KEYMAP = {
    KeyW: "keyForward",
    KeyA: "keyLeft",
    KeyS: "keyBackward",
    KeyD: "keyRight",
    KeyQ: "keyRollCCW",
    KeyE: "keyRollCW",
    Space: "keyUp",
    ShiftLeft: "keyDown"
};

function unpackMarchingCubesTables(buffer) {
    let fileBuffer = new DataView(buffer);
    let readOffset = 0;

    let edgeTable = new Uint32Array(256);
    for (let i = 0; i < edgeTable.length; i++) {
        edgeTable[i] = fileBuffer.getUint16(readOffset, true);
        readOffset += 2;
    }

    let triTable = new Uint32Array(4096);
    for (let i = 0; i < triTable.length; i += 16) {
        let packedIndices = fileBuffer.getBigUint64(readOffset, true);
        let indexCount = Number(packedIndices & 15n);
        triTable[i] = indexCount;
        for (let j = i + 1; j <= i + indexCount; j++) {
            packedIndices >>= 4n;
            triTable[j] = Number(packedIndices & 15n);
        }

        readOffset += 8;
    }

    return [edgeTable, triTable];
}

(async function main() {
    let canvas = document.querySelector("canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    if (!navigator.gpu) {
        alert("WebGPU not supported :(");
        return;
    }

    let adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert("Failed to get WebGPU adapter :(");
        return;
    }

    let device = await adapter.requestDevice({requiredFeatures: ["float32-filterable"]});
    if (!device) {
        alert("Failed to get WebGPU device :(");
        return;
    }

    let [
        rendererShaderCode,
        mcubesShaderCode,
        packedMcubesTables,
        volumeData
    ] = await Promise.all([
        fetch("./shaders/basic_renderer.wgsl").then((response) => response.text()),
        fetch("./shaders/marching_cubes.wgsl").then((response) => response.text()),
        fetch("./assets/mcubes_tables.bin").then((response) => response.arrayBuffer()),
        fetch("./assets/test_volumes/teapot_sdf_256.bin").then((response) => response.arrayBuffer())
    ]);

    let context = canvas.getContext("webgpu");
    let presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: presentationFormat
    });

    let rendererShaderModule = device.createShaderModule({
        label: "Renderer::ShaderModule",
        code: rendererShaderCode
    });

    let marchingCubesModule = device.createShaderModule({
        label: "MarchingCubes::ShaderModule",
        code: mcubesShaderCode
    });

    let renderPipeline = device.createRenderPipeline({
        label: "Renderer::RenderPipeline",
        layout: device.createPipelineLayout({
            bindGroupLayouts: [device.createBindGroupLayout({
                entries: [{
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {type: "uniform"}
                }]
            })]
        }),
        primitive: {
            cullMode: "back",
            frontFace: "ccw",
            topology: "triangle-list"
        },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less"
        },
        vertex: {
            entryPoint: "processVertex",
            module: rendererShaderModule,
            buffers: [{
                arrayStride: 32,
                attributes: [
                    {shaderLocation: 0, offset: 0, format: "float32x3"},
                    {shaderLocation: 1, offset: 16, format: "float32x3"}
                ]
            }]
        },
        fragment: {
            entryPoint: "processFragment",
            module: rendererShaderModule,
            targets: [{format: presentationFormat}]
        }
    });

    let marchingCubesPipeline = device.createComputePipeline({
        label: "MarchingCubes::ComputePipeline",
        layout: device.createPipelineLayout({
            bindGroupLayouts: [device.createBindGroupLayout({
                entries: [{
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "uniform"}
                }, {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: {type: "filtering"}
                }, {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {viewDimension: "3d"}
                }, {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "read-only-storage"}
                }, {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "read-only-storage"}
                }, {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "storage"}
                }, {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "storage"}
                }, {
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "storage"}
                }]
            })]
        }),
        compute: {
            entryPoint: "marchingCubes",
            module: marchingCubesModule
        }
    });

    let volumeSampler = device.createSampler({
        minFilter: "linear",
        magFilter: "linear"
    });

    let volumeTexture = device.createTexture({
        dimension: "3d",
        size: [256, 256, 256],
        format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    device.queue.writeTexture({texture: volumeTexture}, volumeData, {bytesPerRow: 1024, rowsPerImage: 256}, [256, 256, 256]);

    let rendererUniformsBufferCPU = new ArrayBuffer(208);
    let rendererUniformsView = {
        modelMatrix: new Float32Array(rendererUniformsBufferCPU, 0, 16),
        normalMatrix: new Float32Array(rendererUniformsBufferCPU, 64, 16),
        cameraMatrix: new Float32Array(rendererUniformsBufferCPU, 128, 16),
        cameraPosition: new Float32Array(rendererUniformsBufferCPU, 192, 4)
    };

    let rendererUniformsBufferGPU = device.createBuffer({
        label: "Renderer::UniformBuffer",
        size: rendererUniformsBufferCPU.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    let marchingCubesUniformsBufferCPU = new ArrayBuffer(64);
    let marchingCubesUniformsView = {
        marchOrigin: new Float32Array(marchingCubesUniformsBufferCPU, 0, 3),
        marchDimensions: new Float32Array(marchingCubesUniformsBufferCPU, 16, 3),
        marchResolution: new Uint32Array(marchingCubesUniformsBufferCPU, 32, 3),
        gradientSampleDelta: new Float32Array(marchingCubesUniformsBufferCPU, 48, 3),
        isolevel: new Float32Array(marchingCubesUniformsBufferCPU, 60, 1)
    };

    let marchingCubesUniformsBufferGPU = device.createBuffer({
        label: "MarchingCubes::UniformBuffer",
        size: marchingCubesUniformsBufferCPU.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    let vertexStorageBuffer = device.createBuffer({
        label: "MarchingCubes::VertexStorageBuffer",
        size: device.limits.maxStorageBufferBindingSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE
    });

    let indexStorageBuffer = device.createBuffer({
        label: "MarchingCubes::IndexStorageBuffer",
        size: device.limits.maxStorageBufferBindingSize,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE
    });

    let [edgeTable, triTable] = unpackMarchingCubesTables(packedMcubesTables);
    let edgeTableBuffer = device.createBuffer({
        label: "MarchingCubes::EdgeTableBuffer",
        size: edgeTable.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    let triTableBuffer = device.createBuffer({
        label: "MarchingCubes::TriangleTableBuffer",
        size: triTable.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    device.queue.writeBuffer(edgeTableBuffer, 0, edgeTable);
    device.queue.writeBuffer(triTableBuffer, 0, triTable);

    let countersBuffer = device.createBuffer({
        label: "MarchingCubes::CountersBuffer",
        size: 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    let initialCountersBuffer = device.createBuffer({
        label: "MarchingCubes::InitialCountersBuffer",
        size: 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    device.queue.writeBuffer(initialCountersBuffer, 0, new Uint32Array([0, 0]));

    let drawCallBuffer = device.createBuffer({
        label: "Renderer::IndirectDrawCallBuffer",
        size: 20,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
    });

    device.queue.writeBuffer(drawCallBuffer, 0, new Uint32Array([0, 1, 0, 0, 0]));

    let rendererBindGroup = device.createBindGroup({
        label: "Renderer::BindGroup",
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: rendererUniformsBufferGPU}}
        ]
    });

    let marchingCubesBindGroup = device.createBindGroup({
        label: "MarchingCubes::BindGroup",
        layout: marchingCubesPipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: marchingCubesUniformsBufferGPU}},
            {binding: 1, resource: volumeSampler},
            {binding: 2, resource: volumeTexture.createView()},
            {binding: 3, resource: {buffer: edgeTableBuffer}},
            {binding: 4, resource: {buffer: triTableBuffer}},
            {binding: 5, resource: {buffer: vertexStorageBuffer}},
            {binding: 6, resource: {buffer: indexStorageBuffer}},
            {binding: 7, resource: {buffer: countersBuffer}}
        ]
    });

    let renderPassDescriptor = {
        label: "Renderer::RenderPassDescriptor",
        colorAttachments: [{
            clearValue: [0.0, 0.0, 0.0, 1.0],
            loadOp: "clear",
            storeOp: "store"
        }],
        depthStencilAttachment: {
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store"
        }
    };

    let cameraBasis = Matrix4.identity();
    cameraBasis.translate([0.0, 0.0, 4.0]);

    let controlStates = {};
    for (let controlName of Object.values(CAMERA_CONTROL_KEYMAP)) {
        controlStates[controlName] = false;
    }

    window.addEventListener("resize", function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    addEventListener("keydown", function(event) {
        event.preventDefault();
        if (event.code in CAMERA_CONTROL_KEYMAP) {
            controlStates[CAMERA_CONTROL_KEYMAP[event.code]] = true;
        } else if (event.code === "Escape") {
            document.exitPointerLock();
        }
    });

    addEventListener("keyup", function(event) {
        event.preventDefault();
        if (event.code in CAMERA_CONTROL_KEYMAP) {
            controlStates[CAMERA_CONTROL_KEYMAP[event.code]] = false;
        }
    });

    canvas.addEventListener("click", async function(event) {
        event.preventDefault();
        await canvas.requestPointerLock({unadjustedMovement: true});
    });

    canvas.addEventListener("mousemove", function(event) {
        event.preventDefault();
        if (document.pointerLockElement === canvas) {
            cameraBasis.yaw(event.movementX / window.devicePixelRatio * CAMERA_YAW_PITCH_RATE);
            cameraBasis.pitch(event.movementY / window.devicePixelRatio * CAMERA_YAW_PITCH_RATE);
        }
    });

    let stats = new Stats();
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    stats.dom.style.position = "absolute";
    stats.dom.style.zIndex = 2;
    document.getElementById("display").appendChild(stats.dom);

    let depthTexture = null;
    let prevFrameTime = performance.now();
    requestAnimationFrame(function render() {
        stats.begin();

        let canvasTexture = context.getCurrentTexture();
        if (!depthTexture || depthTexture.width != canvasTexture.width || depthTexture.height != canvasTexture.height) {
            if (depthTexture) depthTexture.destroy();
            depthTexture = device.createTexture({
                size: [canvasTexture.width, canvasTexture.height],
                format: "depth24plus",
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });
        }

        renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();
        renderPassDescriptor.depthStencilAttachment.view = depthTexture.createView();
        let encoder = device.createCommandEncoder({label: "Renderer::CommandEncoder"});

        let projMatrix = Matrix4.perspProjection(CAMERA_FOCAL_LENGTH, canvas.width / canvas.height, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE);
        rendererUniformsView.modelMatrix.set(Matrix4.identity());
        rendererUniformsView.normalMatrix.set(Matrix4.identity());
        rendererUniformsView.cameraMatrix.set(projMatrix.transformMatrix(cameraBasis.inverseOrthonormal()));
        rendererUniformsView.cameraPosition.set(cameraBasis.getTranslationComponent());
        device.queue.writeBuffer(rendererUniformsBufferGPU, 0, rendererUniformsBufferCPU);

        marchingCubesUniformsView.marchOrigin.set([-2.0, -2.0, -2.0]);
        marchingCubesUniformsView.marchDimensions.set([4.0, 4.0, 4.0]);
        marchingCubesUniformsView.marchResolution.set([256, 256, 256]);
        marchingCubesUniformsView.gradientSampleDelta.set([0.004, 0.004, 0.004]);
        marchingCubesUniformsView.isolevel.set([0.0]);
        device.queue.writeBuffer(marchingCubesUniformsBufferGPU, 0, marchingCubesUniformsBufferCPU);

        encoder.copyBufferToBuffer(initialCountersBuffer, 0, countersBuffer, 0, 8);
        let marchingCubesPass = encoder.beginComputePass({label: "MarchingCubes::ComputePass"});
        marchingCubesPass.setPipeline(marchingCubesPipeline);
        marchingCubesPass.setBindGroup(0, marchingCubesBindGroup);
        marchingCubesPass.dispatchWorkgroups(64, 64, 64);
        marchingCubesPass.end();

        encoder.copyBufferToBuffer(countersBuffer, 4, drawCallBuffer, 0, 4);
        let renderPass = encoder.beginRenderPass(renderPassDescriptor);
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, rendererBindGroup);
        renderPass.setVertexBuffer(0, vertexStorageBuffer);
        renderPass.setIndexBuffer(indexStorageBuffer, "uint32");
        renderPass.drawIndexedIndirect(drawCallBuffer, 0);
        renderPass.end();

        let commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        let curFrameTime = performance.now();
        let deltaTime = (curFrameTime - prevFrameTime) / 1000.0;
        prevFrameTime = curFrameTime;

        let moveDirX = controlStates.keyRight - controlStates.keyLeft;
        let moveDirY = controlStates.keyUp - controlStates.keyDown;
        let moveDirZ = controlStates.keyBackward - controlStates.keyForward;
        if (moveDirX != 0 || moveDirY != 0 || moveDirZ != 0) {
            let moveRate = CAMERA_MOVEMENT_RATE / Math.hypot(moveDirX, moveDirY, moveDirZ) * deltaTime;
            cameraBasis.translate(cameraBasis.transformDirection([moveDirX * moveRate, moveDirY * moveRate, moveDirZ * moveRate]));
        }

        let rollDir = controlStates.keyRollCCW - controlStates.keyRollCW;
        cameraBasis.roll(rollDir * CAMERA_ROLL_RATE * deltaTime);

        stats.end();
        requestAnimationFrame(render);
    });
})();