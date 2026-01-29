# glTF WebGPU 查看器

[English](./README.md) | [中文](./README_CN.md)

一个基于 **WebGPU** 和 **Rust (WASM)** 构建的高性能 glTF 2.0 查看器。

## 功能特性

-   **WebGPU 后端**：利用现代 WebGPU API 进行高效渲染。
-   **Rust WASM 解析器**：使用 Rust 和 `wasm-bindgen` 实现高性能 glTF 解析。
-   **Draco 压缩支持**：通过 Google Draco Decoder 支持 `KHR_draco_mesh_compression` 扩展。
-   **交互式控制**：支持轨道相机的旋转、平移和缩放。
-   **拖拽上传**：支持直接将 `.gltf`/`.glb` 文件或文件夹拖入窗口。
-   **PBR 渲染**：基础 PBR 光照模型（环境光、方向光、镜面光）。

## 前置要求

在构建之前，请确保已安装以下工具：

-   [Rust](https://www.rust-lang.org/tools/install)
-   [wasm-pack](https://rustwasm.github.io/wasm-pack/installer.html)

```bash
cargo install wasm-pack
```

## 构建指南

1.  克隆仓库：
    ```bash
    git clone https://github.com/your-username/gltf-wgpu-viewer.git
    cd gltf-wgpu-viewer
    ```

2.  构建 WebAssembly 模块：
    ```bash
    wasm-pack build --target web
    ```
    此命令将生成包含编译后的 WASM 和 JS 胶水代码的 `pkg/` 目录。

## 部署 (Netlify)

本项目包含 `netlify.toml` 配置文件，可轻松部署到 Netlify。

1.  将此仓库推送到 GitHub/GitLab/Bitbucket。
2.  登录 [Netlify](https://www.netlify.com/) 并选择 "New site from Git"。
3.  Netlify 会自动检测 `netlify.toml` 设置：
    -   **Build command**: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh && wasm-pack build --target web`
    -   **Publish directory**: `.`

## 本地运行

由于本项目使用 WebGPU 和 WASM，受浏览器安全限制（CORS/模块加载），必须通过本地 HTTP 服务器运行。

你可以使用任何静态文件服务器。例如：

### Python
```bash
python3 -m http.server 8000
# 打开 http://localhost:8000
```

### Node.js (serve)
```bash
npx serve .
# 打开提供的 URL
```

### VS Code
使用 **Live Server** 扩展打开 `index.html`。

## 使用指南

1.  在支持 WebGPU 的浏览器（Chrome 113+、Edge 等）中打开查看器。
2.  点击 **"选择文件夹"** 或 **拖拽** 你的 glTF 文件（`.gltf` + `.bin` + 纹理）或 `.glb` 文件到窗口中。
3.  **控制**：
    -   **左键 + 拖拽**：旋转
    -   **Shift + 左键 + 拖拽**（或右键）：平移
    -   **滚轮**：缩放

## 项目结构

-   `src/lib.rs`: 用于 glTF 解析的 Rust 源代码。
-   `main.js`: 包含 WebGPU 渲染逻辑的主要 JavaScript 入口点。
-   `camera.js`: 相机控制逻辑。
-   `index.html`: UI 和入口页面。
-   `pkg/`: 生成的 WASM 输出（git 已忽略，通过 `wasm-pack` 构建）。

## 许可证

MIT
