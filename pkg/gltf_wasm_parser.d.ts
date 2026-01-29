/* tslint:disable */
/* eslint-disable */

export class MeshData {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly colors: Float32Array;
    readonly draw_calls: Int32Array;
    readonly indices: Uint32Array;
    readonly normals: Float32Array;
    readonly positions: Float32Array;
    readonly uvs: Float32Array;
}

export function parse_multifile_gltf(gltf_filename: string, file_map: any): MeshData;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_meshdata_free: (a: number, b: number) => void;
    readonly meshdata_positions: (a: number) => any;
    readonly meshdata_indices: (a: number) => any;
    readonly meshdata_normals: (a: number) => any;
    readonly meshdata_colors: (a: number) => any;
    readonly meshdata_uvs: (a: number) => any;
    readonly meshdata_draw_calls: (a: number) => any;
    readonly parse_multifile_gltf: (a: number, b: number, c: any) => [number, number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
