/**
 * main.js - WebGPU glTF 查看器核心逻辑
 * 
 * 此文件处理：
 * 1. WebGPU 初始化 (Adapter, Device, Context)。
 * 2. 渲染管线创建 (Shaders, Buffers)。
 * 3. 文件加载与解析 (glTF, GLB, Draco 压缩)。
 * 4. 资源管理 (Textures, Buffers, BindGroups)。
 * 5. 主渲染循环与交互 (Camera, MVP 更新)。
 */

import init, { parse_multifile_gltf } from './pkg/gltf_wasm_parser.js';
import { OrbitCamera } from './camera.js';

// --- Shader 代码 (WGSL) ---
// 定义渲染管线的顶点和片段着色器。
const shaderCode = `
// Uniform 缓冲对象：保存每帧或每次交互更新的全局数据
struct Uniforms {
    mvp: mat4x4<f32>,       // 模型-视图-投影矩阵
    camera_pos: vec3<f32>,  // 相机世界位置（用于镜面光照）
};

// 绑定组 0：Uniforms
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// 绑定组 1：材质 (纹理 + 采样器)
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

// 来自 CPU 缓冲区的顶点属性
struct VertexInput {
    @location(0) pos: vec3<f32>,    // 顶点位置
    @location(1) normal: vec3<f32>, // 顶点法线
    @location(2) color: vec4<f32>,  // 顶点颜色（可选）
    @location(3) uv: vec2<f32>      // 纹理坐标
};

// 顶点着色器输出（传递给片段着色器）
struct vOut {
    @builtin(position) pos: vec4<f32>, // 裁剪空间位置（必须）
    @location(0) normal: vec3<f32>,    // 插值后的法线
    @location(1) color: vec4<f32>,     // 插值后的颜色
    @location(2) uv: vec2<f32>,        // 插值后的 UV
    @location(3) world_pos: vec3<f32>  // 世界空间位置（用于光照）
};

@vertex
fn vs_main(in: VertexInput) -> vOut {
    var out: vOut;
    // 变换位置到裁剪空间
    out.pos = uniforms.mvp * vec4<f32>(in.pos, 1.0);
    // 传递属性
    out.normal = in.normal;
    out.color = in.color;
    out.uv = in.uv;
    out.world_pos = in.pos; // 假设模型矩阵为单位矩阵或已在 CPU 中预应用
    return out;
}

@fragment
fn fs_main(in: vOut) -> @location(0) vec4<f32> {
    // 采样纹理颜色
    let texColor = textureSample(t_diffuse, s_diffuse, in.uv);
    
    // 归一化插值法线
    let n = normalize(in.normal);
    
    // 结合顶点颜色和纹理颜色
    // Fix: 忽略顶点颜色，防止全黑
    let base = texColor.rgb;
    
    // --- 光照计算 ---
    
    // 1. 环境光 (基础照明)
    let ambient = base * 0.3;

    // 视图方向：从片段指向相机的向量
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);

    // 2. 头灯 (跟随相机的方向光)
    // 光照方向与视图方向相同
    let light_dir = view_dir;
    let diff = max(dot(n, light_dir), 0.0);
    let diffuse = base * diff * 0.8;

    // 3. 顶光 (来自上方的固定方向光)
    let top_light_dir = normalize(vec3<f32>(0.2, 1.0, 0.2));
    let diff2 = max(dot(n, top_light_dir), 0.0);
    let diffuse2 = base * diff2 * 0.3;

    // 4. 镜面高光 (Blinn-Phong)
    // 光照和视图之间的半向量。对于头灯，h = view_dir。
    let half_vec = view_dir; 
    let spec_angle = max(dot(n, half_vec), 0.0);
    let specular = pow(spec_angle, 32.0) * 0.2; // 光泽度 = 32.0

    // 组合所有光照分量
    let final_color = ambient + diffuse + diffuse2 + vec3<f32>(specular);

    // Fix: 强制 Alpha 为 1.0
    return vec4<f32>(final_color, 1.0);
}
`;

const dracoBaseUrl = 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/';
let dracoModulePromise = null;

/**
 * 主入口点
 */
async function start() {
    // 初始化 Rust WASM 模块
    await init();
    
    if (!navigator.gpu) return alert("WebGPU 不受支持");

    // --- WebGPU 设置 ---
    // 1. 获取 Adapter (物理 GPU 的抽象)
    // requestAdapter 可能会返回 null，例如在不支持 WebGPU 的浏览器中。
    const adapter = await navigator.gpu.requestAdapter();
    // 2. 获取 Device (逻辑设备)
    // Device 是与 GPU 交互的主要接口，用于创建资源（Buffer, Texture）和管线。
    const device = await adapter.requestDevice();
    
    // 3. 配置 Canvas 上下文
    const canvas = document.getElementById('gpuCanvas');
    const context = canvas.getContext('webgpu');
    // 获取设备首选的 Canvas 格式（通常是 bgra8unorm 或 rgba8unorm）
    const format = navigator.gpu.getPreferredCanvasFormat();

    // 配置上下文以使用设备和格式
    // alphaMode: 'premultiplied' 表示 Canvas 合成时 Alpha 预乘处理，这是 WebGPU 的标准做法。
    context.configure({ device, format, alphaMode: 'premultiplied' });

    // --- 渲染管线创建 ---
    // 渲染管线定义了从顶点输入到像素输出的完整流程。
    // 创建管线是一个昂贵的操作，通常在初始化时完成。
    const pipeline = device.createRenderPipeline({
        layout: 'auto', // 自动推断 BindGroupLayout (Uniforms 和 Textures 的布局)
        vertex: {
            module: device.createShaderModule({ code: shaderCode }),
            entryPoint: 'vs_main', // 顶点着色器入口函数
            // 定义顶点缓冲布局 (Vertex Buffer Layout)
            // 告诉 GPU 如何从内存中读取顶点数据
            buffers: [
                // 槽位 0: 位置 (float32x3 = 12 bytes)
                { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }, 
                // 槽位 1: 法线 (float32x3 = 12 bytes)
                { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }, 
                // 槽位 2: 颜色 (float32x4 = 16 bytes)
                { arrayStride: 16, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x4' }] }, 
                // 槽位 3: UV (float32x2 = 8 bytes)
                { arrayStride: 8, attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x2' }] }  
            ]
        },
        fragment: {
            module: device.createShaderModule({ code: shaderCode }),
            entryPoint: 'fs_main', // 片段着色器入口函数
            targets: [{ format }] // 输出目标的颜色格式（必须匹配 Canvas 格式）
        },
        primitive: { 
            topology: 'triangle-list', // 绘制三角形列表
            cullMode: 'none' // 禁用背面剔除：因为许多 glTF 模型是双面的或法线不一致，禁用剔除最安全
        }, 
        depthStencil: { 
            depthWriteEnabled: true, // 允许写入深度缓冲区
            depthCompare: 'less',    // 深度测试函数：通过如果新深度 < 旧深度
            format: 'depth24plus'    // 深度缓冲区格式
        }
    });

    // --- Uniform 缓冲设置 ---
    // 大小 80: 64 字节 (mat4 MVP) + 16 字节 (vec3 CameraPos + 填充)
    const uniformBuffer = device.createBuffer({
        size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    });

    let vBuffer = null, nBuffer = null, cBuffer = null, uvBuffer = null, iBuffer = null, indexCount = 0;
    let bounds = null;
    let meshData = null;
    let materials = [];
    let defaultMaterial = null;
    let drawCalls = [];
    
    // --- 相机与交互设置 ---
    // 相机状态 (轨道控制)
    const camera = new OrbitCamera(canvas);
    
    // 添加回调以在交互时更新 MVP
    canvas.addEventListener('pointermove', () => {
        if (camera.isDragging || camera.isPanning) {
            updateMVP(device, uniformBuffer, canvas, camera, bounds);
        }
    });
    canvas.addEventListener('wheel', () => {
        updateMVP(device, uniformBuffer, canvas, camera, bounds);
    });

    // --- 文件处理与解析 ---
    // 处理文件选择 (拖放或文件输入)
    const handleUpload = async (files) => {
        const loadStartTime = performance.now();
        console.log("Handle upload:", files);
        const fileMap = {};
        let gltfName = "";
        document.getElementById('status').innerText = '读取文件中...';

        // 1. 将所有文件读取到内存中 (Uint8Array)
        // 使用 Promise.all 并行读取所有文件，提高加载速度。
        await Promise.all(Array.from(files).map(async f => {
            // 如果可用，使用 webkitRelativePath 处理目录结构 (文件夹上传时)
            // 这对于解析 glTF 中引用的相对路径资源至关重要。
            const path = f.webkitRelativePath || f.name;
            // 将路径分隔符规范化为正斜杠，以统一 Windows/Unix 路径
            const normalizedPath = path.replace(/\\/g, '/');
            console.log(`Loading file: ${normalizedPath} (original: ${f.name})`);
            try {
                fileMap[normalizedPath] = new Uint8Array(await f.arrayBuffer());
            } catch (e) {
                console.error(`Failed to read file ${f.name}:`, e);
                return;
            }
            
            // 同时也映射 basename 以便在相对查找失败时进行扁平查找 (Flat Fallback)
            // 例如：如果 glTF 引用 "textures/a.png" 但文件列表中只有 "a.png"，我们可以找到它。
            fileMap[f.name] = fileMap[normalizedPath];

            // 识别入口点 (.gltf 或 .glb)
            // 忽略以 ._ 开头的隐藏文件 (macOS 元数据)
            if ((f.name.endsWith('.gltf') || f.name.endsWith('.glb')) && !f.name.startsWith('._')) {
                gltfName = normalizedPath;
                console.log("Found glTF entry:", gltfName);
            }
        }));

        console.log("FileMap keys:", Object.keys(fileMap));
        if (!gltfName) {
            console.error("No .gltf or .glb file found in upload.");
        }

        try {
            if (!gltfName) {
                // 如果没有有效文件则重置状态
                indexCount = 0;
                vBuffer = null;
                nBuffer = null;
                cBuffer = null;
                uvBuffer = null;
                iBuffer = null;
                document.getElementById('status').innerText = '未选择 .gltf 或 .glb 文件';
                return;
            }

            // 2. 解析网格数据 (Draco 或标准)
            // parseDracoMeshData 同时处理 Draco 压缩和标准 glTF
            const dracoMesh = await parseDracoMeshData(gltfName, fileMap);
            
            // 3. 纹理与材质设置
            // 为没有纹理的网格创建默认材质 (白色纹理)
            // 使用 rgba8unorm-srgb 以匹配标准 glTF sRGB 纹理的预期行为
            const whiteTex = device.createTexture({ size: [1,1], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
            device.queue.writeTexture({ texture: whiteTex }, new Uint8Array([255,255,255,255]), { bytesPerRow: 4 }, { width: 1, height: 1 });
            const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
            
            const createMaterialBindGroup = (texture) => {
                return device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: texture.createView() },
                        { binding: 1, resource: sampler }
                    ]
                });
            };
            
            defaultMaterial = createMaterialBindGroup(whiteTex);

            // 从 glTF 加载材质
            materials = [];
            
            let json = null;
            let buffers = null;
            if (dracoMesh) {
                json = dracoMesh.json;
                buffers = dracoMesh.buffers;
            } else {
                // 回退：如果 dracoMesh 为空 (如果逻辑更新应该不会发生)
                 const parsed = parseGltfAsset(gltfName, fileMap);
                 if (parsed) {
                     json = parsed.json;
                     buffers = parsed.buffers;
                 }
            }

            if (json) {
                // 计算 basePath 用于解析相对图像路径
                const lastSlash = gltfName.replace(/\\/g, '/').lastIndexOf('/');
                const basePath = lastSlash >= 0 ? gltfName.replace(/\\/g, '/').substring(0, lastSlash + 1) : "";

                // 加载图像并创建纹理
                const textures = [];
                if (json.images) {
                    console.log(`Found ${json.images.length} images in glTF`);
                    for (let i = 0; i < json.images.length; i++) {
                        const imgDef = json.images[i];
                        console.log(`Loading image ${i}:`, imgDef);
                        let blob = null;
                        if (imgDef.uri) {
                            // 解析外部图像文件
                            const bytes = resolveFileBytes(fileMap, imgDef.uri, basePath);
                            console.log(`Resolved bytes for ${imgDef.uri}: ${bytes ? bytes.length : 'null'} bytes`);
                            if (bytes) blob = new Blob([bytes], { type: imgDef.mimeType || 'image/png' });
                        } else if (imgDef.bufferView !== undefined) {
                            // 解析嵌入图像 (BufferView)
                            const bv = json.bufferViews[imgDef.bufferView];
                            const buffer = buffers[bv.buffer];
                            const offset = (bv.byteOffset || 0);
                            const length = bv.byteLength;
                            const bytes = buffer.subarray(offset, offset + length);
                            blob = new Blob([bytes], { type: imgDef.mimeType || 'image/png' });
                        }
                        
                        // 从 Blob 创建 WebGPU 纹理
                        if (blob) {
                            try {
                                console.log(`Decoding image ${i}, size: ${blob.size}, type: ${blob.type}`);
                                // createImageBitmap 是异步的，并且比传统的 Image 元素加载更高效，
                                // 因为它可以在后台线程解码图像，且不阻塞主线程。
                                const bitmap = await createImageBitmap(blob);
                                console.log(`Image ${i} decoded: ${bitmap.width}x${bitmap.height}`);
                                const texture = device.createTexture({
                                    size: [bitmap.width, bitmap.height, 1],
                                    format: 'rgba8unorm-srgb',
                                    // 纹理用途：
                                    // TEXTURE_BINDING: 可以在 Shader 中采样
                                    // COPY_DST: 可以从 CPU 或其他纹理复制数据进来
                                    // RENDER_ATTACHMENT: 可以作为渲染目标（虽然这里主要用作贴图）
                                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
                                });
                                // 将解码后的图像数据复制到 GPU 纹理内存中
                                device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
                                textures.push(texture);
                                console.log(`Texture ${i} created successfully`);
                            } catch (e) {
                                console.warn("Failed to load image", i, e);
                                textures.push(whiteTex);
                            }
                        } else {
                            console.warn(`Failed to resolve blob for image ${i}. URI: ${imgDef.uri}, BufferView: ${imgDef.bufferView}`);
                            textures.push(whiteTex);
                        }
                    }
                } else {
                    console.log("No images found in glTF");
                }
                
                // 为每个材质创建 BindGroups
                if (json.materials) {
                    console.log(`Processing ${json.materials.length} materials`);
                    for (let i = 0; i < json.materials.length; i++) {
                        const mat = json.materials[i];
                        let tex = whiteTex;
                        // 检查 baseColorTexture (PBR)
                        let hasTexture = false;
                        if (mat.pbrMetallicRoughness && 
                            mat.pbrMetallicRoughness.baseColorTexture && 
                            json.textures && 
                            json.textures[mat.pbrMetallicRoughness.baseColorTexture.index]) {
                                const texInfo = json.textures[mat.pbrMetallicRoughness.baseColorTexture.index];
                                if (texInfo.source !== undefined && textures[texInfo.source]) {
                                    tex = textures[texInfo.source];
                                    hasTexture = true;
                                    console.log(`Material ${i} using texture ${texInfo.source}`);
                                } else {
                                    console.warn(`Material ${i} texture source ${texInfo.source} missing or invalid`);
                                }
                        }

                        // 如果没有纹理，使用 baseColorFactor 创建纯色纹理
                        if (!hasTexture && mat.pbrMetallicRoughness && mat.pbrMetallicRoughness.baseColorFactor) {
                            const c = mat.pbrMetallicRoughness.baseColorFactor;
                            // glTF 颜色是线性的，我们将其转换为 sRGB 以便在 sRGB 纹理中正确存储，
                            // 或者直接存储并在 Shader 中采样 (sRGB 格式纹理会自动解码为线性)
                            // 这里我们简单映射 [0-1] -> [0-255] 并写入 sRGB 纹理。
                            // 注意：如果 c 是线性空间，写入 sRGB 格式纹理会被视为 sRGB 值。
                            // Shader 读取时会把这个 sRGB 值解码回线性。
                            // 如果 c 已经是线性，我们应该先 LinearToSrgb 再写入，或者使用非 sRGB 格式纹理。
                            // 但为简单起见，且通常导出器可能已经混合了色彩空间，我们先尝试直接写入。
                            // 为了更准确，我们可以做一个简单的 Gamma 0.4545 转换 (Linear -> sRGB)
                            const toSrgb = (v) => Math.pow(v, 1.0/2.2);
                            const r = Math.round(toSrgb(c[0] ?? 1) * 255);
                            const g = Math.round(toSrgb(c[1] ?? 1) * 255);
                            const b = Math.round(toSrgb(c[2] ?? 1) * 255);
                            const a = Math.round((c[3] ?? 1) * 255);
                            
                            // 只有当颜色不是纯白时才创建新纹理 (优化)
                            if (r !== 255 || g !== 255 || b !== 255 || a !== 255) {
                                const colorTex = device.createTexture({ 
                                    size: [1,1], 
                                    format: 'rgba8unorm-srgb', 
                                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST 
                                });
                                device.queue.writeTexture(
                                    { texture: colorTex }, 
                                    new Uint8Array([r, g, b, a]), 
                                    { bytesPerRow: 4 }, 
                                    { width: 1, height: 1 }
                                );
                                tex = colorTex;
                                console.log(`Material ${i} using color factor: [${r},${g},${b},${a}]`);
                            } else {
                                console.log(`Material ${i} using default white (factor is white)`);
                            }
                        } else if (!hasTexture) {
                            console.log(`Material ${i} has no texture and no color factor, using white`);
                        }
                        materials.push(createMaterialBindGroup(tex));
                    }
                }
            }

            // 4. 创建几何缓冲区
            if (dracoMesh) {
                meshData = dracoMesh;
            } else {
                meshData = parse_multifile_gltf(gltfName, fileMap);
            }
            
            const positions = meshData.positions;
            const indices = meshData.indices;
            const normalsData = meshData.normals;
            const colorsData = meshData.colors;
            const uvsData = meshData.uvs;
            
            // 处理属性名称差异 (Rust: draw_calls, JS: drawCalls)
            if (meshData.draw_calls) {
                drawCalls = meshData.draw_calls;
            } else if (meshData.drawCalls) {
                drawCalls = meshData.drawCalls;
            } else {
                drawCalls = [];
            }
            console.log(JSON.stringify({
                vertexCount: positions.length / 3,
                indexCount: indices.length,
                drawCallsCount: drawCalls.length,
                firstDrawCall: drawCalls.slice(0, 9),
                hasUVs: !!uvsData,
                hasColors: !!colorsData
            }));
            console.log("Mesh Data Loaded:", {
                vertexCount: positions.length / 3,
                indexCount: indices.length,
                drawCallsCount: drawCalls.length,
                firstDrawCall: drawCalls.slice(0, 9),
                hasUVs: !!uvsData,
                hasColors: !!colorsData
            });

            // 创建顶点缓冲区
            vBuffer = device.createBuffer({ size: positions.length * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
            new Float32Array(vBuffer.getMappedRange()).set(positions);
            vBuffer.unmap();

            // 创建 UV 缓冲区
            if (uvsData && uvsData.length > 0) {
                uvBuffer = device.createBuffer({ size: uvsData.length * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
                new Float32Array(uvBuffer.getMappedRange()).set(uvsData);
                uvBuffer.unmap();
            } else {
                // 创建全零的默认 UV 缓冲区以防止崩溃
                const vertexCount = positions.length / 3;
                uvBuffer = device.createBuffer({ size: vertexCount * 2 * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
                new Float32Array(uvBuffer.getMappedRange()).fill(0);
                uvBuffer.unmap();
                console.warn("No UVs found, created default zero UV buffer");
            }

            // 创建法线缓冲区 (如果缺失则计算)
            const normals = (normalsData && normalsData.length > 0)
                ? normalsData
                : computeNormals(positions, indices);
            nBuffer = device.createBuffer({ size: normals.length * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
            new Float32Array(nBuffer.getMappedRange()).set(normals);
            nBuffer.unmap();

            // 创建颜色缓冲区
            const colors = (colorsData && colorsData.length > 0)
                ? colorsData
                : defaultColors(positions.length / 3);
            cBuffer = device.createBuffer({ size: colors.length * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
            new Float32Array(cBuffer.getMappedRange()).set(colors);
            cBuffer.unmap();

            // 创建索引缓冲区
            iBuffer = device.createBuffer({ size: indices.length * 4, usage: GPUBufferUsage.INDEX, mappedAtCreation: true });
            new Uint32Array(iBuffer.getMappedRange()).set(indices);
            iBuffer.unmap();
            
            // 计算相机包围盒
            bounds = computeBounds(positions);
            indexCount = indices.length;
            
            // 初始化相机
            const radius = bounds.radius;
            const distance = Math.max(2, (radius / Math.tan(Math.PI / 8)) * 1.5);
            console.log(`Resetting camera: Target=[${bounds.center}], Dist=${distance}, Radius=${radius}`);
            camera.reset(bounds.center, distance);
            
            updateMVP(device, uniformBuffer, canvas, camera, bounds);
            
            // 更新性能统计面板
            const loadEndTime = performance.now();
            document.getElementById('stat-load').innerText = (loadEndTime - loadStartTime).toFixed(2);
            document.getElementById('stat-tris').innerText = (indexCount / 3).toLocaleString();
            document.getElementById('stat-draws').innerText = (drawCalls ? drawCalls.length / 3 : 0).toString();

            document.getElementById('status').innerText = `加载成功: ${indexCount} 索引`;
            if (materials.length > 0) {
                 document.getElementById('status').innerText += ` | 材质: ${materials.length}`;
            }
            window.debugMVP = true; // 在下次更新时触发日志
        } catch (err) {
            // 错误处理
            indexCount = 0;
            vBuffer = null;
            nBuffer = null;
            cBuffer = null;
            uvBuffer = null;
            iBuffer = null;
            bounds = null;
            meshData = null;
            camera.distance = 10; // 重置默认值
            document.getElementById('status').innerText = `加载失败: ${err}`;
            console.error(err);
        }
    };

    document.getElementById('fileInput').onchange = (e) => handleUpload(e.target.files);
    document.getElementById('folderInput').onchange = (e) => handleUpload(e.target.files);

    // --- 拖放支持 (文件夹和文件) ---
    const dropZone = document.body;
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        const items = e.dataTransfer.items;
        const files = [];
        
        // 遍历目录的辅助函数
        const traverseFileTree = async (item, path = "") => {
            if (item.isFile) {
                const file = await new Promise(resolve => item.file(resolve));
                // 手动定义 webkitRelativePath 以保持一致性
                Object.defineProperty(file, 'webkitRelativePath', {
                    value: path + file.name
                });
                files.push(file);
            } else if (item.isDirectory) {
                const dirReader = item.createReader();
                // readEntries 可能不会在一次调用中返回所有条目（罕见但可能），
                // 但为了简单起见，我们假设它有效或用户使用标准的小文件夹。
                // 稳健读取的标准循环：
                const entries = await new Promise((resolve, reject) => {
                    dirReader.readEntries(resolve, reject);
                });
                for (const entry of entries) {
                    await traverseFileTree(entry, path + item.name + "/");
                }
            }
        };

        if (items) {
            console.log("Processing dropped items...");
            for (let i = 0; i < items.length; i++) {
                const item = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
                if (item) {
                    await traverseFileTree(item);
                } else if (items[i].kind === 'file') {
                    // 针对不支持 webkitGetAsEntry 的浏览器的回退
                    const file = items[i].getAsFile();
                    files.push(file);
                }
            }
        }
        
        if (files.length > 0) {
            console.log(`Dropped ${files.length} files.`);
            handleUpload(files);
        }
    });


    // --- 主渲染循环 ---
    // 创建深度纹理，用于深度测试 (Z-buffering)
    let depthTexture = createDepthTexture(device, canvas);
    let firstRender = true;
    
    // FPS 统计变量
    let lastTime = performance.now();
    let frameCount = 0;

    function render() {
        // 如果窗口大小改变，调整画布大小以匹配显示尺寸，避免模糊
        const resized = resizeCanvasToDisplaySize(canvas, context, device, format);
        if (resized) {
            // 画布调整大小后，必须重新创建深度纹理以匹配新尺寸
            depthTexture = createDepthTexture(device, canvas);
            if (bounds) {
                updateMVP(device, uniformBuffer, canvas, camera, bounds);
            }
        }

        // 开始记录 GPU 命令
        // CommandEncoder 用于将所有渲染命令编码为一个命令缓冲区，最后提交给队列执行
        const encoder = device.createCommandEncoder();
        
        // 开始渲染通道 (Render Pass)
        // 描述了渲染的目标（颜色附件、深度附件）以及开始/结束时的操作（清除、存储）
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ 
                view: context.getCurrentTexture().createView(), // 渲染到当前 Canvas 帧
                loadOp: 'clear', // 在渲染前清除颜色缓冲区
                clearValue: [0.1, 0.1, 0.1, 1], // 清除颜色 (深灰色)
                storeOp: 'store' // 渲染后保存结果到纹理（以便显示）
            }],
            depthStencilAttachment: { 
                view: depthTexture.createView(), 
                depthClearValue: 1.0, // 深度缓冲区初始值（最大深度）
                depthLoadOp: 'clear', 
                depthStoreOp: 'store' 
            }
        });

        if (indexCount > 0) {
            // 在第一次渲染时记录统计信息
            if (firstRender) {
                console.log("First Render Stats:", {
                    indexCount,
                    drawCallsCount: drawCalls ? drawCalls.length : 0,
                    materialCount: materials.length,
                    firstMaterial: materials.length > 0 ? "Exists" : "Empty",
                    bounds,
                    camera
                });
                firstRender = false;
            }
            
            // 计算 FPS
            const now = performance.now();
            frameCount++;
            if (now - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (now - lastTime));
                document.getElementById('stat-fps').innerText = fps;
                frameCount = 0;
                lastTime = now;
            }

            // 设置管线和全局 BindGroups
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            
            // 设置顶点缓冲区 (Slots 0-3)
            pass.setVertexBuffer(0, vBuffer);
            pass.setVertexBuffer(1, nBuffer);
            pass.setVertexBuffer(2, cBuffer);
            pass.setVertexBuffer(3, uvBuffer); // UVs
            pass.setIndexBuffer(iBuffer, 'uint32');
            
            // 绘制网格
            if (drawCalls && drawCalls.length > 0) {
                 // 迭代绘制调用 (基于材质的分块)
                 // 格式: [MaterialIndex, IndexStart, IndexCount, ...]
                 for (let i = 0; i < drawCalls.length; i += 3) {
                     const matIndex = drawCalls[i];
                     const start = drawCalls[i+1];
                     const count = drawCalls[i+2];
                     
                     // 绑定材质 (纹理)
                     let matBindGroup = defaultMaterial;
                     if (matIndex >= 0 && matIndex < materials.length) {
                         matBindGroup = materials[matIndex];
                     }
                     pass.setBindGroup(1, matBindGroup);
                     
                     // 发出绘制命令
                     pass.drawIndexed(count, 1, start, 0, 0);
                 }
            } else {
                 // 针对没有绘制调用的单个网格的回退
                 pass.setBindGroup(1, defaultMaterial);
                 pass.drawIndexed(indexCount);
            }
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(render);
    }
    render();
}

/**
 * 更新模型-视图-投影 (MVP) 矩阵并将其上传到 Uniform 缓冲区。
 * 还处理动态近/远平面计算以提高 Z 精度。
 */
function updateMVP(device, buffer, canvas, camera, bounds) {
    const aspect = canvas.width / canvas.height;
    
    // 基于场景包围盒计算近/远平面以提高深度精度
    // 如果没有包围盒，回退到安全默认值 (对于巨大/微小场景可能会导致伪影)
    let near = 0.1;
    let far = 10000.0;
    
    if (bounds && bounds.radius) {
        // 基于半径的启发式
        const r = bounds.radius;
        // 保持近平面尽可能远，但不要在靠近时裁剪物体。
        // 我们将近平面设置为物体半径的 1%。
        near = Math.max(0.01, r * 0.01);
        
        // 远平面: 相机距离 + 半径 * 边距。
        const dist = camera.distance || (r * 3);
        far = Math.max(100.0, dist + r * 10.0);
        
        // 确保远平面相对于近平面不会太小
        if (far < near * 1000) far = near * 1000;
    }

    const projection = mat4.create();
    mat4.perspective(projection, Math.PI / 4, aspect, near, far);
    
    // WebGPU 使用 [0, 1] Z 裁剪空间，而 gl-matrix 使用 [-1, 1]。
    // 此校正矩阵重新映射 Z 范围：
    // OpenGL: [-1, 1] -> WebGPU: [0, 1]
    // Matrix:
    // 1 0 0 0
    // 0 1 0 0
    // 0 0 0.5 0.5  <-- Scale Z by 0.5 and Offset by 0.5
    // 0 0 0 1
    const correction = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,0.5,0, 0,0,0.5,1]);
    mat4.multiply(projection, correction, projection);

    const view = camera.getViewMatrix();
    const eye = camera.getEyePosition();
    
    const target = camera.target;
    
    const mvp = mat4.create();
    // 矩阵乘法顺序: Projection * View
    // 注意：模型变换通常在 Shader 中或作为单独的 Model 矩阵传入，
    // 但在这里对于静态场景，模型矩阵可能已经是单位矩阵或烘焙到顶点数据中。
    mat4.multiply(mvp, projection, view);
    
    // 写入 MVP 到缓冲区 (偏移量 0)
    device.queue.writeBuffer(buffer, 0, mvp);
    // 写入相机位置到缓冲区 (偏移量 64)
    // 64 字节偏移是因为 mat4x4<f32> 占用 16*4 = 64 字节
    device.queue.writeBuffer(buffer, 64, new Float32Array(eye));
    
    // MVP 更新的调试日志 (节流/一次)
    if (window.debugMVP) {
        console.log("MVP Update:", { projection, view, mvp, eye, target, distance: camera.distance, near, far });
        window.debugMVP = false;
    }
}

/**
 * 计算轴对齐包围盒 (AABB) 和包围球半径。
 */
function computeBounds(positions) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i], y = positions[i + 1], z = positions[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }
    const center = [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5];
    const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    const radius = Math.max(dx, dy, dz) * 0.5;
    const bounds = { center, radius };
    console.log(JSON.stringify(bounds));
    console.log("Computed bounds:", bounds);
    return bounds;
}

// 解决包含 .. 和 . 的路径的辅助函数
function resolvePath(base, relative) {
    const stack = base.split('/');
    if (stack[stack.length - 1] === '') stack.pop(); // Remove trailing empty string
    
    const parts = relative.split('/');
    for (const part of parts) {
        if (part === '.' || part === '') continue;
        if (part === '..') {
            if (stack.length > 0) stack.pop();
        } else {
            stack.push(part);
        }
    }
    return stack.join('/');
}

/**
 * 从上传的文件映射中将文件 URI 解析为其二进制内容。
 * 支持精确匹配、相对路径解析、不区分大小写的搜索和深度搜索。
 */
function resolveFileBytes(fileMap, uri, basePath = "") {
    if (uri.startsWith('data:')) {
        return decodeDataUri(uri);
    }
    
    console.log(`Resolving URI: "${uri}" with BasePath: "${basePath}"`);
    const keys = Object.keys(fileMap);

    // 0. 尝试稳健的路径解析
    if (basePath) {
        const normalizedBasePath = basePath.replace(/\\/g, '/');
        const cleanUri = uri.replace(/\\/g, '/');
        const resolvedPath = resolvePath(normalizedBasePath, cleanUri);
        console.log(`  Trying resolved path: "${resolvedPath}"`);
        if (fileMap[resolvedPath]) {
            console.log(`  Match found (resolved path): ${resolvedPath}`);
            return fileMap[resolvedPath];
        }

        // 不区分大小写的解析路径
        const lowerResolvedPath = resolvedPath.toLowerCase();
        for (const key of keys) {
            if (key.toLowerCase() === lowerResolvedPath) {
                console.log(`  Match found (case-insensitive resolved path): ${key}`);
                return fileMap[key];
            }
        }
        
        // 尝试简单连接作为回退
        const cleanUri2 = uri.startsWith('./') ? uri.substring(2) : uri;
        const fullPath = normalizedBasePath + cleanUri2;
        const normalizedFullPath = fullPath.replace(/\\/g, '/');
        if (fileMap[normalizedFullPath]) {
             console.log(`  Match found (simple concat): ${normalizedFullPath}`);
             return fileMap[normalizedFullPath];
        }
    }

    // 1. 精确匹配
    if (fileMap[uri]) {
        console.log(`  Match found (exact): ${uri}`);
        return fileMap[uri];
    }
    
    // 2. 解码匹配
    let decoded = uri;
    try { decoded = decodeURIComponent(uri); } catch {}
    if (fileMap[decoded]) {
        console.log(`  Match found (decoded): ${decoded}`);
        return fileMap[decoded];
    }

    // 3. 规范化路径匹配 (处理 ./ 前缀和反斜杠)
    const normalizedUri = uri.replace(/\\/g, '/');
    const cleanUri = normalizedUri.startsWith('./') ? normalizedUri.substring(2) : normalizedUri;
    if (fileMap[cleanUri]) {
        console.log(`  Match found (normalized): ${cleanUri}`);
        return fileMap[cleanUri];
    }

    // 4. Basename 匹配 (深度搜索)
    // 搜索所有键，查找以 URI 的 basename 结尾的键
    const uriBasename = normalizedUri.split('/').pop();
    const lowerUriBasename = uriBasename.toLowerCase();
    console.log(`  Trying basename deep search: "${uriBasename}"`);
    
    for(const key of keys) {
        const keyBasename = key.split('/').pop();
        if (keyBasename === uriBasename) {
            console.log(`  Match found (basename deep search): ${key}`);
            return fileMap[key];
        }
        if (keyBasename.toLowerCase() === lowerUriBasename) {
            console.log(`  Match found (case-insensitive basename deep search): ${key}`);
            return fileMap[key];
        }
    }

    // 5. 深度路径后缀匹配 (用于子目录中的相对路径)
    for(const key of keys) {
        if(key.endsWith(normalizedUri) || key.endsWith(cleanUri)) {
             console.log(`  Match found (suffix): ${key}`);
             return fileMap[key];
        }
    }
    
    console.warn(`File not found: ${uri} (basePath: ${basePath})`);
    console.log(`Available files:`, keys);
    return null;
}

/**
 * 解析带有 Draco 网格压缩的 glTF。
 * 加载 WASM 解码器，遍历场景图，并解码网格。
 */
async function parseDracoMeshData(gltfName, fileMap) {
    console.log("Starting parseDracoMeshData...");
    const parsed = parseGltfAsset(gltfName, fileMap);
    if (!parsed) {
        console.error("parseGltfAsset returned null");
        return null;
    }
    const { json, buffers } = parsed;
    console.log("GLTF JSON parsed, checking for Draco extension...");

    if (!json || !json.meshes) return null;
    let hasDraco = false;
    let hasNonDraco = false;
    for (const mesh of json.meshes) {
        if (!mesh.primitives) continue;
        for (const primitive of mesh.primitives) {
            const ext = primitive.extensions && primitive.extensions.KHR_draco_mesh_compression;
            if (ext) hasDraco = true;
            else hasNonDraco = true;
        }
    }
    if (!hasDraco) {
        console.log("No Draco extension found in meshes.");
        return null;
    }
    if (hasNonDraco) {
        throw new Error('检测到混合 Draco 和非 Draco primitive，当前仅支持全 Draco 模型');
    }

    console.log("Loading Draco decoder module...");
    const module = await getDracoModule();
    console.log("Draco decoder module loaded. Decoding meshes...");
    
    try {
        // 解码所有网格并扁平化场景图
        const decoded = decodeDracoMeshes(module, json, buffers);
        console.log("Draco meshes decoded successfully.", { 
            positions: decoded.positions.length, 
            drawCalls: decoded.drawCalls.length 
        });
        return createMeshData(decoded.positions, decoded.normals, decoded.colors, decoded.indices, decoded.uvs, decoded.drawCalls);
    } catch (e) {
        console.error("Error in decodeDracoMeshes:", e);
        throw e;
    }
}

function createMeshData(positions, normals, colors, indices, uvs, drawCalls) {
    return {
        positions,
        indices,
        normals,
        colors,
        uvs,
        drawCalls
    };
}

/**
 * 从文件映射解析 glTF 或 GLB 文件。
 * 返回 JSON 对象和二进制缓冲区数组。
 */
function parseGltfAsset(fileName, fileMap) {
    console.log(`Parsing asset: ${fileName}`);
    const fileContent = fileMap[fileName];
    if (!fileContent) {
        console.error(`File not found: ${fileName}`);
        return null;
    }

    let json;
    let buffers = [];

    if (fileName.toLowerCase().endsWith('.glb')) {
        // 解析 GLB
        try {
            const glb = parseGlb(fileContent);
            json = glb.json;
            buffers.push(glb.binaryChunk); // 缓冲 0 是 GLB 二进制块
        } catch (e) {
            console.error("Failed to parse GLB:", e);
            return null;
        }
    } else {
        // 解析 glTF (JSON)
        const textDecoder = new TextDecoder();
        const jsonText = textDecoder.decode(fileContent);
        try {
            json = JSON.parse(jsonText);
        } catch (e) {
            console.error("Failed to parse glTF JSON:", e);
            return null;
        }
        
        // 加载外部缓冲区
        if (json.buffers) {
            // 计算缓冲区的基本路径 (相对于 glTF 文件)
            const lastSlash = fileName.replace(/\\/g, '/').lastIndexOf('/');
            const basePath = lastSlash >= 0 ? fileName.replace(/\\/g, '/').substring(0, lastSlash + 1) : "";

            for (let i = 0; i < json.buffers.length; i++) {
                const bufferDef = json.buffers[i];
                if (bufferDef.uri) {
                     console.log(`Loading buffer ${i}: ${bufferDef.uri}`);
                     const bufferBytes = resolveFileBytes(fileMap, bufferDef.uri, basePath);
                     if (bufferBytes) {
                         buffers.push(bufferBytes);
                     } else {
                         console.error(`Failed to load buffer: ${bufferDef.uri}`);
                         // 推入空缓冲区以避免索引不匹配
                         buffers.push(new Uint8Array(0));
                     }
                } else {
                    // GLB 中的缓冲 0 (在上面处理) 或未定义 URI
                     buffers.push(new Uint8Array(0)); 
                }
            }
        }
    }
    return { json, buffers };
}

function parseGlb(data) {
    const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const magic = dataView.getUint32(0, true);
    if (magic !== 0x46546C67) throw new Error('Invalid GLB magic');
    const version = dataView.getUint32(4, true);
    const length = dataView.getUint32(8, true);
    
    let offset = 12;
    let json = null;
    let binaryChunk = null;

    while (offset < length) {
        const chunkLength = dataView.getUint32(offset, true);
        const chunkType = dataView.getUint32(offset + 4, true);
        offset += 8;

        if (chunkType === 0x4E4F534A) { // JSON
            const jsonBytes = data.subarray(offset, offset + chunkLength);
            const textDecoder = new TextDecoder();
            json = JSON.parse(textDecoder.decode(jsonBytes));
        } else if (chunkType === 0x004E4942) { // BIN
            binaryChunk = data.subarray(offset, offset + chunkLength);
        }
        
        offset += chunkLength;
    }
    return { json, binaryChunk };
}

function decodeDataUri(uri) {
    const comma = uri.indexOf(',');
    if (comma === -1) return null;
    const base64 = uri.substring(comma + 1);
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

// ...

/**
 * Loads the Draco Decoder WASM module.
 */
async function getDracoModule() {
    if (dracoModulePromise) return dracoModulePromise;
    dracoModulePromise = new Promise((resolve, reject) => {
        // ... (Draco loader logic)
        // Check if global DracoDecoderModule exists or load script
        if (window.DracoDecoderModule) {
            try {
                const module = DracoDecoderModule({
                    locateFile: (file) => `${dracoBaseUrl}${file}`
                });
                if (module && typeof module.then === 'function') {
                    module.then(resolve).catch(reject);
                } else {
                    resolve(module);
                }
            } catch (err) {
                reject(err);
            }
            return;
        }
        const script = document.createElement('script');
        script.src = `${dracoBaseUrl}draco_wasm_wrapper.js`;
        script.onload = () => {
            try {
                const module = DracoDecoderModule({
                    locateFile: (file) => `${dracoBaseUrl}${file}`
                });
                if (module && typeof module.then === 'function') {
                    module.then(resolve).catch(reject);
                } else {
                    resolve(module);
                }
            } catch (err) {
                reject(err);
            }
        };
        script.onerror = () => reject(new Error('加载 Draco 解码器失败'));
        document.head.appendChild(script);
    });
    return dracoModulePromise;
}

/**
 * 遍历 glTF 场景图并解码 Draco 网格。
 * 将节点变换 (matrix/TRS) 应用于顶点。
 */
function decodeDracoMeshes(module, json, buffers) {
    console.log("decodeDracoMeshes started");
    if (typeof mat4 === 'undefined') {
        throw new Error("gl-matrix mat4 未定义！请确保已加载 gl-matrix.js。");
    }
    if (typeof vec3 === 'undefined') {
        throw new Error("gl-matrix vec3 未定义！请确保已加载 gl-matrix.js。");
    }

    const finalPositions = [];
    const finalNormals = [];
    const finalColors = [];
    const finalUvs = [];
    const finalIndices = [];
    // 步长 3: [material_index, index_start, index_count]
    const finalDrawCalls = [];
    let vertexOffset = 0;
    
    // 缓存已解码的网格: meshIndex -> [primitiveData...]
    const meshCache = new Map();

    function getDecodedMesh(meshIndex) {
        if (meshCache.has(meshIndex)) return meshCache.get(meshIndex);
        
        console.log(`Decoding mesh index ${meshIndex}...`);
        const mesh = json.meshes[meshIndex];
        const primitivesData = [];
        
        for (const primitive of mesh.primitives) {
            const dracoExt = primitive.extensions.KHR_draco_mesh_compression;
            // 使用现有逻辑解码 primitive
            const decoded = decodeDracoPrimitive(module, json, buffers, primitive, dracoExt);
            primitivesData.push(decoded);
        }
        
        meshCache.set(meshIndex, primitivesData);
        return primitivesData;
    }

    function getNodeMatrix(node, out) {
        if (node.matrix) {
            mat4.copy(out, node.matrix);
        } else {
            const t = node.translation || [0, 0, 0];
            const r = node.rotation || [0, 0, 0, 1]; // 四元数
            const s = node.scale || [1, 1, 1];
            mat4.fromRotationTranslationScale(out, r, t, s);
        }
    }

    // 递归场景遍历
    // 深度优先遍历场景图，计算每个节点的全局变换矩阵 (World Matrix)
    function traverse(nodeIndex, parentMatrix) {
        // console.log(`Traversing node ${nodeIndex}`); // 太频繁？
        const node = json.nodes[nodeIndex];
        const localMatrix = mat4.create();
        getNodeMatrix(node, localMatrix);
        
        // 计算世界矩阵：World = ParentWorld * Local
        const worldMatrix = mat4.create();
        mat4.multiply(worldMatrix, parentMatrix, localMatrix);
        
        if (node.mesh !== undefined) {
            console.log(`Node ${nodeIndex} has mesh ${node.mesh}`);
            const primitives = getDecodedMesh(node.mesh);
            for (const prim of primitives) {
                const posCount = prim.positions.length / 3;
                
                // 法线矩阵：世界矩阵的逆转置 (Inverse Transpose)
                // 当模型进行非均匀缩放时，法线不能直接使用世界矩阵变换，否则会不再垂直于表面。
                // 必须使用世界矩阵的逆转置矩阵来变换法线以保持其垂直性。
                const normalMatrix = mat4.create();
                mat4.invert(normalMatrix, worldMatrix);
                mat4.transpose(normalMatrix, normalMatrix);

                // 将位置变换到世界空间
                for (let i = 0; i < posCount; i++) {
                    const p = vec3.fromValues(
                        prim.positions[i*3],
                        prim.positions[i*3+1],
                        prim.positions[i*3+2]
                    );
                    vec3.transformMat4(p, p, worldMatrix);
                    finalPositions.push(p[0], p[1], p[2]);
                }

                // 将法线变换到世界空间
                if (prim.normals && prim.normals.length > 0) {
                    for (let i = 0; i < posCount; i++) {
                        const n = vec3.fromValues(
                            prim.normals[i*3],
                            prim.normals[i*3+1],
                            prim.normals[i*3+2]
                        );
                        vec3.transformMat4(n, n, normalMatrix);
                        vec3.normalize(n, n);
                        finalNormals.push(n[0], n[1], n[2]);
                    }
                }

                // 复制颜色（或默认为白色）
                if (prim.colors && prim.colors.length > 0) {
                     for (let i = 0; i < prim.colors.length; i++) finalColors.push(prim.colors[i]);
                } else {
                    for(let i=0; i<posCount; i++) finalColors.push(1,1,1,1);
                }

                // 复制 UV（或默认为 0）
                if (prim.uvs && prim.uvs.length > 0) {
                     for (let i = 0; i < prim.uvs.length; i++) finalUvs.push(prim.uvs[i]);
                } else {
                    for(let i=0; i<posCount; i++) finalUvs.push(0, 0);
                }

                // 偏移索引
                const indexStart = finalIndices.length;
                for (let i = 0; i < prim.indices.length; i++) {
                    finalIndices.push(prim.indices[i] + vertexOffset);
                }
                const indexCount = finalIndices.length - indexStart;

                // 记录绘制调用
                const matIndex = prim.material !== undefined ? prim.material : -1;
                finalDrawCalls.push(matIndex, indexStart, indexCount);
                
                vertexOffset += posCount;
            }
        }
        
        if (node.children) {
            for (const childIndex of node.children) {
                traverse(childIndex, worldMatrix);
            }
        }
    }

    const sceneIndex = json.scene !== undefined ? json.scene : 0;
    const scene = json.scenes ? json.scenes[sceneIndex] : undefined;
    console.log(`Scene index: ${sceneIndex}, Scene found: ${!!scene}`);
    
    if (scene && scene.nodes) {
        console.log(`Starting traversal from scene nodes: ${scene.nodes}`);
        const rootMatrix = mat4.create(); // Identity
        for (const nodeIndex of scene.nodes) {
            traverse(nodeIndex, rootMatrix);
        }
    } else {
        console.warn("No scene/nodes found, falling back to flat mesh iteration");
        // 回退：如果没有场景图，则在原点解码所有网格（旧版行为）
        const meshes = json.meshes || [];
        for (let i = 0; i < meshes.length; i++) {
            const primitives = getDecodedMesh(i);
            for (const prim of primitives) {
                const posCount = prim.positions.length / 3;
                for (let k = 0; k < prim.positions.length; k++) finalPositions.push(prim.positions[k]);
                if (prim.normals) for (let k = 0; k < prim.normals.length; k++) finalNormals.push(prim.normals[k]);
                if (prim.colors) for (let k = 0; k < prim.colors.length; k++) finalColors.push(prim.colors[k]);
                else for(let k=0; k<posCount; k++) finalColors.push(1,1,1,1);
                if (prim.uvs) for (let k = 0; k < prim.uvs.length; k++) finalUvs.push(prim.uvs[k]);
                else for(let k=0; k<posCount; k++) finalUvs.push(0, 0);
                const indexStart = finalIndices.length;
                for (let k = 0; k < prim.indices.length; k++) finalIndices.push(prim.indices[k] + vertexOffset);
                const indexCount = finalIndices.length - indexStart;
                
                // 记录绘制调用（回退）
                const matIndex = prim.material !== undefined ? prim.material : -1;
                finalDrawCalls.push(matIndex, indexStart, indexCount);

                vertexOffset += posCount;
            }
        }
    }

    console.log("Traveral complete. Total vertices:", finalPositions.length / 3);
    const posArray = new Float32Array(finalPositions);
    const nrmArray = finalNormals.length > 0 ? new Float32Array(finalNormals) : new Float32Array();
    const colArray = new Float32Array(finalColors);
    const uvsArray = new Float32Array(finalUvs);
    const idxArray = new Uint32Array(finalIndices);
    const dcArray = new Int32Array(finalDrawCalls);
    return { positions: posArray, normals: nrmArray, colors: colArray, uvs: uvsArray, indices: idxArray, drawCalls: dcArray };
}

/**
 * 解码单个 Draco Primitive。
 * 使用 Google Draco WASM 解码器将压缩数据转换为 Float32Array/Uint32Array。
 */
function decodeDracoPrimitive(module, json, buffers, primitive, dracoExt) {
    const bufferView = json.bufferViews[dracoExt.bufferView];
    const bufferBytes = buffers[bufferView.buffer];
    const byteOffset = bufferView.byteOffset || 0;
    const byteLength = bufferView.byteLength || 0;
    
    // 获取压缩数据的子数组
    const dracoData = bufferBytes.subarray(byteOffset, byteOffset + byteLength);
    
    // 创建解码器实例
    const decoder = new module.Decoder();
    const buffer = new module.DecoderBuffer();
    // 初始化解码缓冲区 (必须传入字节长度)
    buffer.Init(dracoData, dracoData.length);
    
    // 验证几何类型 (必须是三角网格)
    const geometryType = decoder.GetEncodedGeometryType(buffer);
    if (geometryType !== module.TRIANGULAR_MESH) {
        module.destroy(buffer);
        module.destroy(decoder);
        throw new Error('Draco 几何类型不是三角网格');
    }
    
    // 解码到 Mesh 对象
    const mesh = new module.Mesh();
    const status = decoder.DecodeBufferToMesh(buffer, mesh);
    module.destroy(buffer); // 缓冲区不再需要，释放内存
    
    if (!status.ok() || mesh.num_points() === 0) {
        const msg = status.error_msg ? status.error_msg() : 'Draco 解码失败';
        module.destroy(mesh);
        module.destroy(decoder);
        throw new Error(msg);
    }
    
    // 获取面数并创建索引数组
    const numFaces = mesh.num_faces();
    const idxArray = new Uint32Array(numFaces * 3);
    const ia = new module.DracoInt32Array(); // 用于读取索引的临时 C++ 数组对象
    
    for (let i = 0; i < numFaces; i++) {
        decoder.GetFaceFromMesh(mesh, i, ia);
        const offset = i * 3;
        idxArray[offset] = ia.GetValue(0);
        idxArray[offset + 1] = ia.GetValue(1);
        idxArray[offset + 2] = ia.GetValue(2);
    }
    module.destroy(ia); // 释放临时数组
    
    const attributes = dracoExt.attributes || {};
    // ... 读取属性 ...
    if (attributes.POSITION === undefined) {
        module.destroy(mesh);
        module.destroy(decoder);
        throw new Error('Draco 缺少 POSITION 属性');
    }
    const positions = readDracoAttribute(module, decoder, mesh, attributes.POSITION);
    let normals = null;
    if (attributes.NORMAL !== undefined) {
        normals = readDracoAttribute(module, decoder, mesh, attributes.NORMAL);
    }
    let colors = null;
    if (attributes.COLOR_0 !== undefined) {
        colors = readDracoAttribute(module, decoder, mesh, attributes.COLOR_0);
    }
    let uvs = null;
    if (attributes.TEXCOORD_0 !== undefined) {
        uvs = readDracoAttribute(module, decoder, mesh, attributes.TEXCOORD_0);
    } else if (attributes.TEX_COORD_0 !== undefined) { // 针对潜在非标准命名的回退
        uvs = readDracoAttribute(module, decoder, mesh, attributes.TEX_COORD_0);
    }
    
    console.log(`Draco Primitive Decoded: Points=${mesh.num_points()}, Faces=${mesh.num_faces()}, Attributes=`, Object.keys(attributes));
    if (uvs) console.log(`  UVs found: ${uvs.length} floats`);
    else console.log(`  No UVs found`);
    const vertexCount = mesh.num_points();
    const baseColor = getBaseColor(json, primitive);
    if (!colors || colors.length === 0) {
        colors = new Float32Array(vertexCount * 4);
        for (let i = 0; i < vertexCount; i++) {
            const offset = i * 4;
            colors[offset] = baseColor[0];
            colors[offset + 1] = baseColor[1];
            colors[offset + 2] = baseColor[2];
            colors[offset + 3] = baseColor[3];
        }
    } else if (colors.length === vertexCount * 3) {
        const expanded = new Float32Array(vertexCount * 4);
        for (let i = 0; i < vertexCount; i++) {
            const src = i * 3;
            const dst = i * 4;
            expanded[dst] = colors[src];
            expanded[dst + 1] = colors[src + 1];
            expanded[dst + 2] = colors[src + 2];
            expanded[dst + 3] = 1.0;
        }
        colors = expanded;
    }
    module.destroy(mesh);
    module.destroy(decoder);
    // decodeDracoMeshes 内部使用的 Primitive 数据结构
    return { positions, normals, colors, uvs, indices: idxArray, material: primitive.material };
}

function readDracoAttribute(module, decoder, mesh, attributeId) {
    const attribute = decoder.GetAttributeByUniqueId(mesh, attributeId);
    const numComponents = attribute.num_components();
    const numPoints = mesh.num_points();
    const numValues = numPoints * numComponents;
    const dracoArray = new module.DracoFloat32Array();
    decoder.GetAttributeFloatForAllPoints(mesh, attribute, dracoArray);
    const out = new Float32Array(numValues);
    for (let i = 0; i < numValues; i++) {
        out[i] = dracoArray.GetValue(i);
    }
    module.destroy(dracoArray);
    return out;
}

function getBaseColor(json, primitive) {
    if (primitive.material !== undefined && json.materials && json.materials[primitive.material]) {
        const mat = json.materials[primitive.material];
        if (mat.pbrMetallicRoughness && Array.isArray(mat.pbrMetallicRoughness.baseColorFactor)) {
            const c = mat.pbrMetallicRoughness.baseColorFactor;
            return [c[0] ?? 1, c[1] ?? 1, c[2] ?? 1, c[3] ?? 1];
        }
    }
    return [1, 1, 1, 1];
}

function computeNormals(positions, indices) {
    const normals = new Float32Array(positions.length);
    for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i] * 3;
        const i1 = indices[i + 1] * 3;
        const i2 = indices[i + 2] * 3;
        const ax = positions[i1] - positions[i0];
        const ay = positions[i1 + 1] - positions[i0 + 1];
        const az = positions[i1 + 2] - positions[i0 + 2];
        const bx = positions[i2] - positions[i0];
        const by = positions[i2 + 1] - positions[i0 + 1];
        const bz = positions[i2 + 2] - positions[i0 + 2];
        const nx = ay * bz - az * by;
        const ny = az * bx - ax * bz;
        const nz = ax * by - ay * bx;
        normals[i0] += nx;
        normals[i0 + 1] += ny;
        normals[i0 + 2] += nz;
        normals[i1] += nx;
        normals[i1 + 1] += ny;
        normals[i1 + 2] += nz;
        normals[i2] += nx;
        normals[i2 + 1] += ny;
        normals[i2 + 2] += nz;
    }
    for (let i = 0; i < normals.length; i += 3) {
        const x = normals[i], y = normals[i + 1], z = normals[i + 2];
        const len = Math.hypot(x, y, z) || 1;
        normals[i] = x / len;
        normals[i + 1] = y / len;
        normals[i + 2] = z / len;
    }
    return normals;
}

function defaultColors(vertexCount) {
    const colors = new Float32Array(vertexCount * 4);
    for (let i = 0; i < vertexCount; i++) {
        const offset = i * 4;
        colors[offset] = 0.2;
        colors[offset + 1] = 0.75;
        colors[offset + 2] = 1.0;
        colors[offset + 3] = 1.0;
    }
    return colors;
}

function resizeCanvasToDisplaySize(canvas, context, device, format) {
    const pixelRatio = window.devicePixelRatio || 1;
    const width = Math.floor(canvas.clientWidth * pixelRatio);
    const height = Math.floor(canvas.clientHeight * pixelRatio);
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        context.configure({ device, format, alphaMode: 'premultiplied' });
        return true;
    }
    return false;
}

function createDepthTexture(device, canvas) {
    return device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
}

start().catch(err => {
    const statusEl = document.getElementById('status');
    if (statusEl) statusEl.innerText = `初始化失败: ${err}`;
    console.error(err);
});
