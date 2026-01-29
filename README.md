# glTF WebGPU Viewer

[English](./README.md) | [中文](./README_CN.md)

A high-performance glTF 2.0 viewer built with **WebGPU** and **Rust (WASM)**.

## Features

-   **WebGPU Backend**: Utilizes the modern WebGPU API for efficient rendering.
-   **Rust WASM Parser**: High-performance glTF parsing using Rust and `wasm-bindgen`.
-   **Draco Compression**: Supports `KHR_draco_mesh_compression` extension via Google Draco Decoder.
-   **Interactive Controls**: Orbit camera with rotation, panning, and zooming.
-   **Drag & Drop**: Supports dragging `.gltf`/`.glb` files or folders directly into the window.
-   **PBR Rendering**: Basic PBR lighting model (Ambient, Directional, Specular).

## Prerequisites

Before building, ensure you have the following installed:

-   [Rust](https://www.rust-lang.org/tools/install)
-   [wasm-pack](https://rustwasm.github.io/wasm-pack/installer.html)

```bash
cargo install wasm-pack
```

## Build Instructions

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/gltf-wgpu-viewer.git
    cd gltf-wgpu-viewer
    ```

2.  Build the WebAssembly module:
    ```bash
    wasm-pack build --target web
    ```
    This command will generate the `pkg/` directory containing the compiled WASM and JS glue code.

## Deployment (Netlify)

This project includes a `netlify.toml` configuration for easy deployment.

1.  Push this repository to GitHub/GitLab/Bitbucket.
2.  Log in to [Netlify](https://www.netlify.com/) and create a "New site from Git".
3.  Netlify will detect the `netlify.toml` settings automatically:
    -   **Build command**: `echo 'Using pre-built pkg directory'` (Skipped as `pkg/` is included)
    -   **Publish directory**: `.`

## Running Locally

Since this project uses WebGPU and WASM, it must be served via a local HTTP server due to browser security restrictions (CORS/Module loading).

You can use any static file server. Examples:

### Python
```bash
python3 -m http.server 8000
# Open http://localhost:8000
```

### Node.js (serve)
```bash
npx serve .
# Open the provided URL
```

### VS Code
Use the **Live Server** extension to open `index.html`.

## Usage

1.  Open the viewer in a WebGPU-compatible browser (Chrome 113+, Edge, etc.).
2.  Click **"Choose Folder"** or **Drag & Drop** your glTF files (`.gltf` + `.bin` + textures) or `.glb` files into the window.
3.  **Controls**:
    -   **Left Click + Drag**: Rotate
    -   **Shift + Left Click + Drag** (or Right Click): Pan
    -   **Scroll**: Zoom

## Project Structure

-   `src/lib.rs`: Rust source code for glTF parsing.
-   `main.js`: Main JavaScript entry point containing WebGPU rendering logic.
-   `camera.js`: Camera control logic.
-   `index.html`: UI and entry point.
-   `pkg/`: Generated WASM output (excluded from git, built via `wasm-pack`).

## License

MIT
