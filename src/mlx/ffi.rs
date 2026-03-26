//! Raw FFI bindings to the MLX C library (libmlxc).
//!
//! These declarations map directly to the mlx-c headers:
//!   https://github.com/ml-explore/mlx-c/tree/main/mlx/c
//!
//! All functions follow the same pattern:
//!   - Output is written through a `*mut mlx_array` pointer (first arg)
//!   - The last arg is always the `mlx_stream` to execute on
//!   - Caller owns the returned array and must free it with `mlx_array_free`
//!
//! Safety: all functions are unsafe. Use the safe wrappers in `ops.rs`.

use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Opaque handle types (foreign C structs — never dereferenced in Rust)
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct mlx_array_t(c_void);
pub type mlx_array = *mut mlx_array_t;

#[repr(C)]
pub struct mlx_stream_t(c_void);
pub type mlx_stream = *mut mlx_stream_t;

#[repr(C)]
pub struct mlx_device_t(c_void);
pub type mlx_device = *mut mlx_device_t;

/// Container for a variable-length list of mlx_arrays (used by concatenate etc.)
#[repr(C)]
pub struct mlx_vector_array_t(c_void);
pub type mlx_vector_array = *mut mlx_vector_array_t;

// ---------------------------------------------------------------------------
// Data type enum  (matches mlx/c/array.h)
// ---------------------------------------------------------------------------
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mlx_dtype {
    bool_     = 0,
    uint8     = 1,
    uint16    = 2,
    uint32    = 3,
    uint64    = 4,
    int8      = 5,
    int16     = 6,
    int32     = 7,
    int64     = 8,
    float16   = 9,
    float32   = 10,
    bfloat16  = 11,
    complex64 = 12,
}

// ---------------------------------------------------------------------------
// Device type
// ---------------------------------------------------------------------------
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum mlx_device_type {
    cpu = 0,
    gpu = 1,
}

#[link(name = "mlxc")]
extern "C" {
    // -----------------------------------------------------------------------
    // Array lifecycle
    // -----------------------------------------------------------------------

    /// Create an empty (null) array handle.
    pub fn mlx_array_new() -> mlx_array;

    /// Free an array and release its memory.
    pub fn mlx_array_free(arr: mlx_array);

    /// Create a 0-d scalar float array.
    pub fn mlx_array_new_float(val: f32) -> mlx_array;

    /// Create an array from a flat float32 buffer with given shape.
    /// `data` must remain valid until `mlx_eval` is called.
    pub fn mlx_array_new_data(
        data: *const c_void,
        shape: *const i32,
        dim: i32,
        dtype: mlx_dtype,
    ) -> mlx_array;

    // -----------------------------------------------------------------------
    // Evaluation (MLX uses lazy execution — call eval before reading data)
    // -----------------------------------------------------------------------

    /// Force evaluation of a single array on its stream.
    pub fn mlx_array_eval(arr: mlx_array);

    // -----------------------------------------------------------------------
    // Shape / metadata queries
    // -----------------------------------------------------------------------

    pub fn mlx_array_ndim(arr: mlx_array) -> usize;
    pub fn mlx_array_dim(arr: mlx_array, dim: i32) -> i32;
    pub fn mlx_array_size(arr: mlx_array) -> usize;
    pub fn mlx_array_dtype(arr: mlx_array) -> mlx_dtype;

    // -----------------------------------------------------------------------
    // Data access (only valid after mlx_array_eval)
    // -----------------------------------------------------------------------

    pub fn mlx_array_data_float32(arr: mlx_array) -> *const f32;
    pub fn mlx_array_data_int32(arr: mlx_array) -> *const i32;
    pub fn mlx_array_data_int64(arr: mlx_array) -> *const i64;

    // -----------------------------------------------------------------------
    // Stream / device
    // -----------------------------------------------------------------------

    pub fn mlx_default_stream(device: mlx_device) -> mlx_stream;
    pub fn mlx_default_device() -> mlx_device;
    pub fn mlx_set_default_device(device: mlx_device);
    pub fn mlx_device_new_type(device_type: mlx_device_type, index: i32) -> mlx_device;
    pub fn mlx_device_free(device: mlx_device);
    pub fn mlx_stream_free(stream: mlx_stream);
    pub fn mlx_stream_synchronize(stream: mlx_stream);

    // -----------------------------------------------------------------------
    // vector_array (for ops that take a list of arrays)
    // -----------------------------------------------------------------------

    pub fn mlx_vector_array_new() -> mlx_vector_array;
    pub fn mlx_vector_array_free(vec: mlx_vector_array);
    pub fn mlx_vector_array_append_value(vec: mlx_vector_array, arr: mlx_array);
    pub fn mlx_vector_array_size(vec: mlx_vector_array) -> usize;

    // -----------------------------------------------------------------------
    // Shape manipulation
    // -----------------------------------------------------------------------

    pub fn mlx_reshape(
        res: *mut mlx_array,
        arr: mlx_array,
        shape: *const i32,
        ndim: usize,
        stream: mlx_stream,
    );

    pub fn mlx_flatten(
        res: *mut mlx_array,
        arr: mlx_array,
        start_axis: i32,
        end_axis: i32,
        stream: mlx_stream,
    );

    pub fn mlx_transpose_axes(
        res: *mut mlx_array,
        arr: mlx_array,
        axes: *const i32,
        num_axes: usize,
        stream: mlx_stream,
    );

    pub fn mlx_squeeze(
        res: *mut mlx_array,
        arr: mlx_array,
        axes: *const i32,
        num_axes: usize,
        stream: mlx_stream,
    );

    pub fn mlx_expand_dims(
        res: *mut mlx_array,
        arr: mlx_array,
        axes: *const i32,
        num_axes: usize,
        stream: mlx_stream,
    );

    pub fn mlx_slice(
        res: *mut mlx_array,
        arr: mlx_array,
        start: *const i32,
        stop: *const i32,
        strides: *const i32,
        num_axes: usize,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Concatenation / stacking
    // -----------------------------------------------------------------------

    pub fn mlx_concatenate_axis(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        axis: i32,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    pub fn mlx_add(
        res: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    );

    pub fn mlx_subtract(
        res: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    );

    pub fn mlx_multiply(
        res: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    );

    pub fn mlx_divide(
        res: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    );

    pub fn mlx_negative(
        res: *mut mlx_array,
        a: mlx_array,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Linear algebra
    // -----------------------------------------------------------------------

    pub fn mlx_matmul(
        res: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    );

    pub fn mlx_addmm(
        res: *mut mlx_array,
        c: mlx_array,
        a: mlx_array,
        b: mlx_array,
        alpha: f32,
        beta: f32,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Reductions
    // -----------------------------------------------------------------------

    pub fn mlx_sum_axes(
        res: *mut mlx_array,
        arr: mlx_array,
        axes: *const i32,
        num_axes: usize,
        keepdims: bool,
        stream: mlx_stream,
    );

    pub fn mlx_mean_axes(
        res: *mut mlx_array,
        arr: mlx_array,
        axes: *const i32,
        num_axes: usize,
        keepdims: bool,
        stream: mlx_stream,
    );

    pub fn mlx_argmax_axis(
        res: *mut mlx_array,
        arr: mlx_array,
        axis: i32,
        keepdims: bool,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Activations
    // -----------------------------------------------------------------------

    pub fn mlx_relu(res: *mut mlx_array, a: mlx_array, stream: mlx_stream);
    pub fn mlx_silu(res: *mut mlx_array, a: mlx_array, stream: mlx_stream);
    pub fn mlx_gelu(res: *mut mlx_array, a: mlx_array, stream: mlx_stream);
    pub fn mlx_tanh(res: *mut mlx_array, a: mlx_array, stream: mlx_stream);
    pub fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, stream: mlx_stream);

    pub fn mlx_softmax_axes(
        res: *mut mlx_array,
        arr: mlx_array,
        axes: *const i32,
        num_axes: usize,
        precise: bool,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Normalisation (fast path in mlx.core.fast)
    // -----------------------------------------------------------------------

    /// Fast fused layer norm.
    /// weight and bias may be null for no-affine variant.
    pub fn mlx_fast_layer_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        bias: mlx_array,
        eps: f32,
        stream: mlx_stream,
    );

    pub fn mlx_fast_rms_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        eps: f32,
        stream: mlx_stream,
    );

    /// Scaled dot-product attention.
    /// mask may be null.
    pub fn mlx_fast_scaled_dot_product_attention(
        res: *mut mlx_array,
        queries: mlx_array,
        keys: mlx_array,
        values: mlx_array,
        scale: f32,
        mask: mlx_array,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Indexing / gathering
    // -----------------------------------------------------------------------

    /// Equivalent to arr[indices] along axis 0.
    pub fn mlx_take(
        res: *mut mlx_array,
        arr: mlx_array,
        indices: mlx_array,
        axis: i32,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Type conversion
    // -----------------------------------------------------------------------

    pub fn mlx_astype(
        res: *mut mlx_array,
        arr: mlx_array,
        dtype: mlx_dtype,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Creation helpers
    // -----------------------------------------------------------------------

    pub fn mlx_zeros(
        res: *mut mlx_array,
        shape: *const i32,
        ndim: usize,
        dtype: mlx_dtype,
        stream: mlx_stream,
    );

    pub fn mlx_ones(
        res: *mut mlx_array,
        shape: *const i32,
        ndim: usize,
        dtype: mlx_dtype,
        stream: mlx_stream,
    );

    pub fn mlx_arange(
        res: *mut mlx_array,
        start: f64,
        stop: f64,
        step: f64,
        dtype: mlx_dtype,
        stream: mlx_stream,
    );

    pub fn mlx_full(
        res: *mut mlx_array,
        shape: *const i32,
        ndim: usize,
        vals: mlx_array,
        dtype: mlx_dtype,
        stream: mlx_stream,
    );

    // -----------------------------------------------------------------------
    // Convolutions
    //   MLX uses channels-last (NHWC) layout:
    //     conv2d input: (N, H, W, C)   weight: (out_C, kH, kW, in_C/groups)
    //     conv1d input: (N, L, C)       weight: (out_C, kW, in_C/groups)
    // PyTorch weights from safetensors are NCHW/OIHW — caller must transpose.
    // -----------------------------------------------------------------------

    /// 2-D convolution (no built-in bias — add separately).
    pub fn mlx_conv2d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        groups: i32,
        stream: mlx_stream,
    );

    /// 1-D convolution (no built-in bias — add separately).
    pub fn mlx_conv1d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        stream: mlx_stream,
    );
}
