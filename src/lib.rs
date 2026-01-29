use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Uint32Array, Uint8Array};
use std::collections::HashMap;

/// MeshData 结构体：用于存储解析后的网格数据
/// 这些数据将被传递给 JavaScript 端的 WebGPU 进行渲染
#[wasm_bindgen]
pub struct MeshData {
    positions: Vec<f32>, // 顶点位置数据 (x, y, z)
    indices: Vec<u32>,   // 顶点索引数据
    normals: Vec<f32>,   // 顶点法线数据 (nx, ny, nz)
    colors: Vec<f32>,    // 顶点颜色数据 (r, g, b, a)
    uvs: Vec<f32>,       // 纹理坐标数据 (u, v)
    // 绘制调用信息
    // 步长为 3: [material_index, index_start, index_count]
    // 分别代表：材质索引、索引起始位置、索引数量
    draw_calls: Vec<i32>,
}

#[wasm_bindgen]
impl MeshData {
    // 获取位置数据的 getter，返回 Float32Array 给 JS
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.positions) }
    }

    // 获取索引数据的 getter，返回 Uint32Array 给 JS
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Uint32Array {
        unsafe { Uint32Array::view(&self.indices) }
    }

    // 获取法线数据的 getter，返回 Float32Array 给 JS
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.normals) }
    }

    // 获取颜色数据的 getter，返回 Float32Array 给 JS
    #[wasm_bindgen(getter)]
    pub fn colors(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.colors) }
    }

    // 获取 UV 数据的 getter，返回 Float32Array 给 JS
    #[wasm_bindgen(getter)]
    pub fn uvs(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.uvs) }
    }

    // 获取绘制调用信息的 getter，返回 Int32Array 给 JS
    #[wasm_bindgen(getter)]
    pub fn draw_calls(&self) -> js_sys::Int32Array {
        unsafe { js_sys::Int32Array::view(&self.draw_calls) }
    }
}

/// 辅助函数：从 JS 对象（文件映射）中获取文件内容
fn get_js_file(file_map: &JsValue, key: &str) -> Option<Vec<u8>> {
    let key_js = JsValue::from_str(key);
    let val = js_sys::Reflect::get(file_map, &key_js).ok()?;
    if val.is_undefined() || val.is_null() {
        return None;
    }
    let arr = Uint8Array::new(&val);
    Some(arr.to_vec())
}

/// 辅助函数：处理文件路径解析并从 JS 侧获取文件
/// 支持 URL 解码、相对路径处理等多种尝试方式
fn fetch_file_from_js(file_map: &JsValue, uri: &str) -> Option<Vec<u8>> {
    // 1. 尝试直接使用 URI 获取
    if let Some(v) = get_js_file(file_map, uri) {
        return Some(v);
    }
    // 2. 尝试 URL 解码后获取
    let decoded = urlencoding::decode(uri)
        .map(|v| v.into_owned())
        .unwrap_or_else(|_| uri.to_string());
    if let Some(v) = get_js_file(file_map, decoded.as_str()) {
        return Some(v);
    }
    // 3. 去除 "./" 前缀后尝试
    let uri_trim = uri.strip_prefix("./").unwrap_or(uri);
    if let Some(v) = get_js_file(file_map, uri_trim) {
        return Some(v);
    }
    // 4. 去除 "./" 前缀且解码后尝试
    let decoded_trim = decoded.strip_prefix("./").unwrap_or(decoded.as_str());
    if let Some(v) = get_js_file(file_map, decoded_trim) {
        return Some(v);
    }
    // 5. 仅使用文件名（去除路径）尝试
    let uri_base = uri_trim.rsplit(|c| c == '/' || c == '\\').next().unwrap_or(uri_trim);
    if let Some(v) = get_js_file(file_map, uri_base) {
        return Some(v);
    }
    // 6. 仅使用解码后的文件名尝试
    let decoded_base = decoded_trim
        .rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or(decoded_trim);
    if let Some(v) = get_js_file(file_map, decoded_base) {
        return Some(v);
    }
    None
}

/// 返回 4x4 单位矩阵
fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// 4x4 矩阵乘法
fn mat_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = 
                a[0][r] * b[c][0] +
                a[1][r] * b[c][1] +
                a[2][r] * b[c][2] +
                a[3][r] * b[c][3];
        }
    }
    out
}

/// 变换点坐标 (x, y, z) -> (x', y', z')
fn transform_point(m: &[[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    let x = v[0]; let y = v[1]; let z = v[2];
    let w = 1.0;
    [
        m[0][0]*x + m[1][0]*y + m[2][0]*z + m[3][0]*w,
        m[0][1]*x + m[1][1]*y + m[2][1]*z + m[3][1]*w,
        m[0][2]*x + m[1][2]*y + m[2][2]*z + m[3][2]*w,
    ]
}

/// 变换法线向量
/// 注意：这里简化处理，仅使用旋转部分（适用于统一缩放的情况）
fn transform_normal(m: &[[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    let x = v[0]; let y = v[1]; let z = v[2];
    // 近似：仅使用矩阵的旋转部分
    let nx = m[0][0]*x + m[1][0]*y + m[2][0]*z;
    let ny = m[0][1]*x + m[1][1]*y + m[2][1]*z;
    let nz = m[0][2]*x + m[1][2]*y + m[2][2]*z;
    
    // 归一化结果
    let len = (nx*nx + ny*ny + nz*nz).sqrt();
    if len > 0.0 {
        [nx/len, ny/len, nz/len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// 主解析函数：解析多文件 glTF
/// gltf_filename: 入口文件名
/// file_map: JS 对象，包含所有相关文件的二进制数据
#[wasm_bindgen]
pub fn parse_multifile_gltf(gltf_filename: &str, file_map: JsValue) -> Result<MeshData, JsValue> {
    // 设置 panic 钩子以便在控制台显示 Rust 错误
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    if gltf_filename.is_empty() {
        return Err(JsValue::from_str("No .gltf/.glb selected"));
    }
    
    // 1. 获取主 .gltf 文件内容
    let gltf_content = get_js_file(&file_map, gltf_filename).ok_or("No .gltf found")?;
    
    // 2. 解析 GLTF JSON 结构
    let gltf = gltf::Gltf::from_slice(&gltf_content).map_err(|e| e.to_string())?;

    // 3. 预加载引用的 buffer 文件
    let mut files: HashMap<String, Vec<u8>> = HashMap::new();
    for buffer in gltf.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source() {
            if let Some(data) = fetch_file_from_js(&file_map, uri) {
                files.insert(uri.to_string(), data);
            }
        }
    }

    // 初始化数据容器
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut uvs = Vec::new();
    let mut draw_calls = Vec::new();
    let mut normals_complete = true;
    let mut global_vertex_offset = 0u32;

    // 使用栈进行场景图遍历（处理节点层级和变换）
    let mut stack: Vec<(gltf::Node, [[f32; 4]; 4])> = Vec::new();
    
    // 从默认场景或第一个场景开始遍历
    if let Some(scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
        for node in scene.nodes() {
            stack.push((node, identity()));
        }
    } else {
        // 如果没有场景，理论上可以遍历所有 mesh，但这在有效 glTF 中很少见
        // 这里留空，视作无需处理
    }

    // 深度优先遍历节点
    while let Some((node, parent_mat)) = stack.pop() {
        // 计算当前节点的世界变换矩阵 = 父节点矩阵 * 本地矩阵
        let local_mat = node.transform().matrix();
        let world_mat = mat_mul(&parent_mat, &local_mat);
        
        // 如果节点包含 mesh，则处理 mesh 数据
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| {
                    match buffer.source() {
                        gltf::buffer::Source::Uri(uri) => {
                            files.get(uri).map(|v| v.as_slice())
                        }
                        gltf::buffer::Source::Bin => gltf.blob.as_deref(),
                    }
                });

                // 读取并变换顶点位置
                let mut prim_positions = Vec::new();
                if let Some(pos_iter) = reader.read_positions() {
                    for p in pos_iter {
                        prim_positions.push(transform_point(&world_mat, p));
                    }
                }
                
                let vertex_count = prim_positions.len() as u32;
                if vertex_count == 0 { continue; }

                // 追加位置数据
                for p in &prim_positions {
                    positions.extend_from_slice(p);
                }

                // 处理索引
                let index_start = indices.len() as u32;
                if let Some(idx) = reader.read_indices() {
                    // 如果有索引 buffer，需要加上当前的全局顶点偏移
                    indices.extend(idx.into_u32().map(|i| i + global_vertex_offset));
                } else {
                    // 如果没有索引，生成连续的索引
                    indices.extend((0..vertex_count).map(|i| i + global_vertex_offset));
                }
                let index_count = (indices.len() as u32) - index_start;
                
                // 记录 DrawCall 信息
                let mat_idx = primitive.material().index().map(|i| i as i32).unwrap_or(-1);
                draw_calls.push(mat_idx);
                draw_calls.push(index_start as i32);
                draw_calls.push(index_count as i32);

                // 处理法线
                if normals_complete {
                    if let Some(nrm_iter) = reader.read_normals() {
                        for n in nrm_iter {
                            normals.extend_from_slice(&transform_normal(&world_mat, n));
                        }
                    } else {
                        // 如果某个 primitive 缺失法线，标记整体不完整
                        normals_complete = false;
                        normals.clear();
                    }
                }

                // 处理颜色
                let material = primitive.material();
                let base_color = material.pbr_metallic_roughness().base_color_factor();
                
                match reader.read_colors(0) {
                    Some(gltf::mesh::util::ReadColors::RgbF32(iter)) => {
                        for c in iter { colors.extend_from_slice(&[c[0], c[1], c[2], 1.0]); }
                    }
                    Some(gltf::mesh::util::ReadColors::RgbaF32(iter)) => {
                        for c in iter { colors.extend_from_slice(&c); }
                    }
                    Some(gltf::mesh::util::ReadColors::RgbU8(iter)) => {
                        for c in iter { colors.extend_from_slice(&[c[0] as f32/255.0, c[1] as f32/255.0, c[2] as f32/255.0, 1.0]); }
                    }
                    Some(gltf::mesh::util::ReadColors::RgbaU8(iter)) => {
                        for c in iter { colors.extend_from_slice(&[c[0] as f32/255.0, c[1] as f32/255.0, c[2] as f32/255.0, c[3] as f32/255.0]); }
                    }
                    Some(gltf::mesh::util::ReadColors::RgbU16(iter)) => {
                        for c in iter { colors.extend_from_slice(&[c[0] as f32/65535.0, c[1] as f32/65535.0, c[2] as f32/65535.0, 1.0]); }
                    }
                    Some(gltf::mesh::util::ReadColors::RgbaU16(iter)) => {
                        for c in iter { colors.extend_from_slice(&[c[0] as f32/65535.0, c[1] as f32/65535.0, c[2] as f32/65535.0, c[3] as f32/65535.0]); }
                    }
                    None => {
                        // 如果没有顶点颜色，使用材质的基础颜色填充
                        for _ in 0..vertex_count { colors.extend_from_slice(&base_color); }
                    }
                }

                // 处理 UV 坐标
                match reader.read_tex_coords(0) {
                    Some(gltf::mesh::util::ReadTexCoords::F32(iter)) => {
                        for uv in iter { uvs.extend_from_slice(&uv); }
                    }
                    Some(gltf::mesh::util::ReadTexCoords::U8(iter)) => {
                        for uv in iter { uvs.extend_from_slice(&[uv[0] as f32/255.0, uv[1] as f32/255.0]); }
                    }
                    Some(gltf::mesh::util::ReadTexCoords::U16(iter)) => {
                        for uv in iter { uvs.extend_from_slice(&[uv[0] as f32/65535.0, uv[1] as f32/65535.0]); }
                    }
                    None => {
                        // 如果没有 UV，填充 0
                        for _ in 0..vertex_count { uvs.extend_from_slice(&[0.0, 0.0]); }
                    }
                }
                
                // 更新全局顶点偏移
                global_vertex_offset += vertex_count;
            }
        }
        
        // 将子节点加入栈中继续遍历
        for child in node.children() {
            stack.push((child, world_mat));
        }
    }

    // 检查是否有数据
    if positions.is_empty() {
        return Err(JsValue::from_str("No positions found (or empty scene)"));
    }
    if !normals_complete {
        normals.clear();
    }
    // 返回构建好的 MeshData
    Ok(MeshData { positions, indices, normals, colors, uvs, draw_calls })
}
