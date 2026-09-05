#include "CLTensor.h"
#include "utils.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <dlprim/core/activation.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace coeiroink::opencl {

using at::Scalar;
using at::ScalarType;
using at::Tensor;

namespace {

// VITS推論で必要な索引演算だけをOpenCL上で実行し、範囲外アクセスはinvalidフラグ経由でホストへ通知する。
constexpr char kIndexSource[] = R"CLC(
__kernel void gather_value(
    __global const VALUE_TYPE *input,
    ulong input_offset,
    __global const long *indices,
    ulong index_offset,
    __global VALUE_TYPE *output,
    ulong output_offset,
    ulong total,
    ulong inner,
    ulong input_dim,
    ulong output_dim,
    int wrap_negative,
    __global uint *invalid) {
    ulong linear = get_global_id(0);
    if (linear >= total) return;
    ulong block = output_dim * inner;
    ulong outer = linear / block;
    ulong rem = linear % block;
    long selected = indices[index_offset + linear];
    if (selected < 0 && wrap_negative) selected += (long)input_dim;
    if (selected < 0 || (ulong)selected >= input_dim) {
        atomic_or((volatile __global uint *)invalid, 1u);
        return;
    }
    ulong source = outer * input_dim * inner + (ulong)selected * inner + rem % inner;
    output[output_offset + linear] = input[input_offset + source];
}

__kernel void index_select_value(
    __global const VALUE_TYPE *input,
    ulong input_offset,
    __global const long *indices,
    ulong index_offset,
    __global VALUE_TYPE *output,
    ulong output_offset,
    ulong total,
    ulong inner,
    ulong input_dim,
    ulong index_count,
    int wrap_negative,
    __global uint *invalid) {
    ulong linear = get_global_id(0);
    if (linear >= total) return;
    ulong block = index_count * inner;
    ulong outer = linear / block;
    ulong rem = linear % block;
    long selected = indices[index_offset + rem / inner];
    if (selected < 0 && wrap_negative) selected += (long)input_dim;
    if (selected < 0 || (ulong)selected >= input_dim) {
        atomic_or((volatile __global uint *)invalid, 1u);
        return;
    }
    ulong source = outer * input_dim * inner + (ulong)selected * inner + rem % inner;
    output[output_offset + linear] = input[input_offset + source];
}

__kernel void flip_value(
    __global const VALUE_TYPE *input,
    ulong input_offset,
    __global VALUE_TYPE *output,
    ulong output_offset,
    ulong total,
    int rank,
    ulong flip_mask,
    ulong d0, ulong d1, ulong d2, ulong d3,
    ulong d4, ulong d5, ulong d6, ulong d7) {
    ulong linear = get_global_id(0);
    if (linear >= total) return;
    ulong dims[8] = {d0, d1, d2, d3, d4, d5, d6, d7};
    ulong remaining = linear;
    ulong stride = 1;
    ulong source = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        ulong coordinate = remaining % dims[axis];
        remaining /= dims[axis];
        if ((flip_mask & ((ulong)1 << axis)) != 0) {
            coordinate = dims[axis] - coordinate - 1;
        }
        source += coordinate * stride;
        stride *= dims[axis];
    }
    output[output_offset + linear] = input[input_offset + source];
}

__kernel void count_true(
    __global const uchar *mask,
    ulong mask_offset,
    ulong mask_size,
    __global uint *count) {
    if (get_global_id(0) != 0) return;
    uint value = 0;
    for (ulong index = 0; index < mask_size; ++index) {
        value += mask[mask_offset + index] != 0;
    }
    count[0] = value;
}

__kernel void masked_index_value(
    __global const VALUE_TYPE *input,
    ulong input_offset,
    __global const uchar *mask,
    ulong mask_offset,
    __global VALUE_TYPE *output,
    ulong output_offset,
    ulong mask_size,
    ulong inner_size) {
    if (get_global_id(0) != 0) return;
    ulong selected = 0;
    for (ulong mask_index = 0; mask_index < mask_size; ++mask_index) {
        if (mask[mask_offset + mask_index] == 0) continue;
        ulong source = mask_index * inner_size;
        ulong target = selected * inner_size;
        for (ulong inner = 0; inner < inner_size; ++inner) {
            output[output_offset + target + inner] = input[input_offset + source + inner];
        }
        ++selected;
    }
}

__kernel void masked_index_put_value(
    __global VALUE_TYPE *target,
    ulong target_offset,
    __global const uchar *mask,
    ulong mask_offset,
    __global const VALUE_TYPE *values,
    ulong values_offset,
    ulong values_size,
    ulong mask_size,
    ulong inner_size) {
    if (get_global_id(0) != 0) return;
    ulong selected = 0;
    for (ulong mask_index = 0; mask_index < mask_size; ++mask_index) {
        if (mask[mask_offset + mask_index] == 0) continue;
        ulong target_row = mask_index * inner_size;
        ulong value_row = selected * inner_size;
        for (ulong inner = 0; inner < inner_size; ++inner) {
            ulong value_index = values_size == 1 ? 0 : value_row + inner;
            target[target_offset + target_row + inner] = values[values_offset + value_index];
        }
        ++selected;
    }
}

__kernel void nonzero_bool(
    __global const uchar *input,
    ulong input_offset,
    __global long *output,
    ulong output_offset,
    ulong total,
    int rank,
    ulong d0, ulong d1, ulong d2, ulong d3,
    ulong d4, ulong d5, ulong d6, ulong d7) {
    if (get_global_id(0) != 0) return;
    ulong dims[8] = {d0, d1, d2, d3, d4, d5, d6, d7};
    ulong selected = 0;
    for (ulong linear = 0; linear < total; ++linear) {
        if (input[input_offset + linear] == 0) continue;
        ulong remaining = linear;
        for (int axis = rank - 1; axis >= 0; --axis) {
            output[output_offset + selected * rank + axis] = (long)(remaining % dims[axis]);
            remaining /= dims[axis];
        }
        ++selected;
    }
}
)CLC";

// VITSが利用する最終軸の縮約とweight normalizationに範囲を限定したOpenCLカーネル群。
constexpr char kReductionSource[] = R"CLC(
__kernel void cumsum_last_float(
    __global const float *input,
    ulong input_offset,
    __global float *output,
    ulong output_offset,
    ulong lines,
    ulong width) {
    ulong line = get_global_id(0);
    if (line >= lines) return;
    ulong base = line * width;
    float sum = 0.0f;
    for (ulong column = 0; column < width; ++column) {
        sum += input[input_offset + base + column];
        output[output_offset + base + column] = sum;
    }
}

__kernel void max_last_float(
    __global const float *input,
    ulong input_offset,
    __global float *values,
    ulong values_offset,
    __global long *indices,
    ulong indices_offset,
    ulong lines,
    ulong width) {
    ulong line = get_global_id(0);
    if (line >= lines) return;
    ulong base = line * width;
    float best = input[input_offset + base];
    long best_index = 0;
    for (ulong column = 1; column < width; ++column) {
        float candidate = input[input_offset + base + column];
        if ((isnan(candidate) && !isnan(best)) ||
            (!isnan(candidate) && !isnan(best) && candidate > best)) {
            best = candidate;
            best_index = (long)column;
        }
    }
    values[values_offset + line] = best;
    indices[indices_offset + line] = best_index;
}

__kernel void all_bool(
    __global const uchar *input,
    ulong input_offset,
    __global uchar *output,
    ulong output_offset,
    ulong total) {
    if (get_global_id(0) != 0) return;
    uchar result = 1;
    for (ulong index = 0; index < total; ++index) {
        if (input[input_offset + index] == 0) {
            result = 0;
            break;
        }
    }
    output[output_offset] = result;
}

__kernel void weight_norm_dim0_float(
    __global const float *v,
    ulong v_offset,
    __global const float *g,
    ulong g_offset,
    __global float *weight,
    ulong weight_offset,
    __global float *norms,
    ulong norms_offset,
    ulong groups,
    ulong group_size) {
    ulong group = get_global_id(0);
    if (group >= groups) return;
    ulong base = group * group_size;
    float sum = 0.0f;
    for (ulong item = 0; item < group_size; ++item) {
        float value = v[v_offset + base + item];
        sum += value * value;
    }
    float norm = sqrt(sum);
    float scale = g[g_offset + group] / norm;
    norms[norms_offset + group] = norm;
    for (ulong item = 0; item < group_size; ++item) {
        weight[weight_offset + base + item] = v[v_offset + base + item] * scale;
    }
}
)CLC";

int64_t normalize_dim(int64_t dim, int64_t rank) {
    TORCH_CHECK(rank > 0, "OpenCL operation requires a non-scalar tensor");
    if (dim < 0) dim += rank;
    TORCH_CHECK(dim >= 0 && dim < rank, "dimension out of range");
    return dim;
}

int64_t product(at::IntArrayRef sizes, int64_t begin, int64_t end) {
    int64_t result = 1;
    for (int64_t index = begin; index < end; ++index) {
        TORCH_CHECK(
            sizes[index] == 0 || result <= std::numeric_limits<int64_t>::max() / sizes[index],
            "tensor size overflow");
        result *= sizes[index];
    }
    return result;
}

std::vector<int64_t> shape(at::IntArrayRef sizes) {
    return {sizes.begin(), sizes.end()};
}

Tensor new_tensor(
    const std::vector<int64_t>& sizes,
    const Tensor& reference,
    ScalarType dtype) {
    return ptdlprim::new_ocl_tensor(sizes, reference.device(), dtype);
}

Tensor contiguous_on_device(
    const Tensor& tensor,
    const Tensor& reference,
    ScalarType dtype) {
    Tensor value = tensor;
    if (value.device() != reference.device() || value.scalar_type() != dtype) {
        value = value.to(reference.device(), dtype);
    }
    return value.contiguous();
}

std::string opencl_type(ScalarType dtype) {
    return dlprim::data_type_to_opencl_type(ptdlprim::todp(dtype), true);
}

void register_sources() {
    // dlprimのプロセス共通キャッシュへ一度だけ登録し、同名の異なるソースを暗黙に上書きしない。
    static std::once_flag once;
    std::call_once(once, [] {
        auto [index_source, index_inserted] =
            dlprim::gpu::kernel_sources.emplace("coeiroink_vits_index", kIndexSource);
        TORCH_CHECK(
            index_inserted || index_source->second == kIndexSource,
            "OpenCL kernel source collision: coeiroink_vits_index");
        auto [reduction_source, reduction_inserted] =
            dlprim::gpu::kernel_sources.emplace("coeiroink_vits_reduction", kReductionSource);
        TORCH_CHECK(
            reduction_inserted || reduction_source->second == kReductionSource,
            "OpenCL kernel source collision: coeiroink_vits_reduction");
    });
}

const cl::Program& program(
    const Tensor& reference,
    const char* source,
    std::vector<dlprim::gpu::Parameter> parameters = {}) {
    register_sources();
    dlprim::Context context(ptdlprim::getExecutionContext(reference));
    return dlprim::gpu::Cache::instance().get_program(context, source, parameters);
}

void enqueue(const Tensor& reference, cl::Kernel& kernel, uint64_t count, const char* name) {
    if (count == 0) return;
    auto execution = ptdlprim::getExecutionContext(reference);
    execution.queue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(count)),
        cl::NullRange,
        execution.events(),
        execution.event(name));
}

cl::Buffer buffer(const Tensor& tensor) {
    return ptdlprim::buffer_from_tensor(tensor);
}

Tensor new_error_flag(const Tensor& reference) {
    Tensor flag = new_tensor({1}, reference, at::kInt);
    uint32_t zero = 0;
    auto execution = ptdlprim::getExecutionContext(reference);
    execution.queue().enqueueWriteBuffer(
        buffer(flag), CL_TRUE, 0, sizeof(zero), &zero);
    return flag;
}

void check_error_flag(
    const Tensor& reference,
    const Tensor& flag,
    const char* message) {
    // OpenCLカーネル内では例外を送出できないため、索引検証時だけ小さなフラグを同期的に読み出す。
    uint32_t value = 0;
    auto execution = ptdlprim::getExecutionContext(reference);
    execution.queue().enqueueReadBuffer(
        buffer(flag), CL_TRUE, 0, sizeof(value), &value);
    TORCH_CHECK(value == 0, message);
}

uint32_t count_true_values(const Tensor& mask) {
    // bool索引の出力形状はデータ依存なので、GPUで数えた要素数だけを同期的に読み出す。
    TORCH_CHECK(mask.scalar_type() == at::kBool, "OpenCL mask must use bool");
    TORCH_CHECK(
        mask.numel() <= std::numeric_limits<uint32_t>::max(),
        "OpenCL mask is too large");
    Tensor contiguous = mask.contiguous();
    auto allocation = ptdlprim::CLContextManager::allocate(mask.device(), sizeof(cl_uint));
    cl::Buffer counter(static_cast<cl_mem>(allocation.get()), true);
    const auto& compiled = program(
        mask,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", "uchar")});
    cl::Kernel kernel(compiled, "count_true");
    int argument = 0;
    kernel.setArg(argument++, buffer(contiguous));
    kernel.setArg(argument++, static_cast<cl_ulong>(contiguous.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(contiguous.numel()));
    kernel.setArg(argument++, counter);
    enqueue(mask, kernel, 1, "coeiroink_count_true");

    uint32_t count = 0;
    auto execution = ptdlprim::getExecutionContext(mask);
    execution.queue().enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(count), &count);
    return count;
}

std::array<cl_ulong, 8> padded_dimensions(at::IntArrayRef sizes) {
    TORCH_CHECK(sizes.size() <= 8, "OpenCL backend supports at most 8 dimensions");
    std::array<cl_ulong, 8> dimensions{1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t index = 0; index < sizes.size(); ++index) {
        dimensions[index] = static_cast<cl_ulong>(sizes[index]);
    }
    return dimensions;
}

Tensor single_index(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
    // 現行VITSが使う軸0の単一索引だけを専用実装し、この演算内の不正・未対応な索引形式は明示的に拒否する。
    TORCH_CHECK(!indices.empty(), "OpenCL indexing requires one index tensor");
    TORCH_CHECK(indices.get(0).has_value(), "OpenCL indexing requires an index at axis 0");
    for (size_t axis = 1; axis < indices.size(); ++axis) {
        TORCH_CHECK(!indices.get(axis).has_value(), "OpenCL VITS indexing supports only axis 0");
    }
    Tensor index = *indices.get(0);
    TORCH_CHECK(index.device() == self.device(), "index tensor must use the input device");
    return index;
}

void copy_to_out(const Tensor& value, Tensor& out) {
    TORCH_CHECK(out.device() == value.device(), "output must use the input device");
    TORCH_CHECK(out.scalar_type() == value.scalar_type(), "output dtype mismatch");
    at::assert_no_internal_overlap(out);
    if (!out.sizes().equals(value.sizes())) out.resize_(value.sizes());
    out.copy_(value);
}

}  // namespace

Tensor& gather_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& out) {
    TORCH_CHECK(!sparse_grad, "OpenCL gather does not support sparse gradients");
    dim = normalize_dim(dim, self.dim());
    TORCH_CHECK(index.scalar_type() == at::kLong, "gather index must use int64");
    TORCH_CHECK(index.device() == self.device(), "gather index must use the input device");
    TORCH_CHECK(index.dim() == self.dim(), "gather index rank must match the input rank");
    for (int64_t axis = 0; axis < self.dim(); ++axis) {
        if (axis != dim) {
            TORCH_CHECK(index.size(axis) == self.size(axis), "gather shape mismatch");
        }
    }
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_overlap(out, index);
    if (!out.sizes().equals(index.sizes())) out.resize_(index.sizes());
    TORCH_CHECK(out.scalar_type() == self.scalar_type(), "gather output dtype mismatch");
    if (out.numel() == 0) return out;

    Tensor input = self.contiguous();
    Tensor indices = index.contiguous();
    Tensor output = out.contiguous();
    Tensor error_flag = new_error_flag(self);
    const auto& compiled = program(
        self,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", opencl_type(self.scalar_type()))});
    cl::Kernel kernel(compiled, "gather_value");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(indices));
    kernel.setArg(argument++, static_cast<cl_ulong>(indices.storage_offset()));
    kernel.setArg(argument++, buffer(output));
    kernel.setArg(argument++, static_cast<cl_ulong>(output.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(output.numel()));
    kernel.setArg(argument++, static_cast<cl_ulong>(product(self.sizes(), dim + 1, self.dim())));
    kernel.setArg(argument++, static_cast<cl_ulong>(self.size(dim)));
    kernel.setArg(argument++, static_cast<cl_ulong>(index.size(dim)));
    kernel.setArg(argument++, static_cast<cl_int>(0));
    kernel.setArg(argument++, buffer(error_flag));
    enqueue(self, kernel, output.numel(), "coeiroink_gather");
    check_error_flag(self, error_flag, "gather index out of range");
    if (!out.is_contiguous()) out.copy_(output);
    return out;
}

Tensor gather(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad) {
    Tensor out = new_tensor(shape(index.sizes()), self, self.scalar_type());
    gather_out(self, dim, index, sparse_grad, out);
    return out;
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
    dim = normalize_dim(dim, self.dim());
    TORCH_CHECK(index.scalar_type() == at::kLong, "index_select index must use int64");
    TORCH_CHECK(index.device() == self.device(), "index_select index must use the input device");
    TORCH_CHECK(index.dim() <= 1, "index_select index must be a scalar or vector");
    std::vector<int64_t> sizes = shape(self.sizes());
    sizes[dim] = index.numel();
    Tensor out = new_tensor(sizes, self, self.scalar_type());
    if (out.numel() == 0) return out;

    Tensor input = self.contiguous();
    Tensor indices = index.contiguous();
    Tensor error_flag = new_error_flag(self);
    const auto& compiled = program(
        self,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", opencl_type(self.scalar_type()))});
    cl::Kernel kernel(compiled, "index_select_value");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(indices));
    kernel.setArg(argument++, static_cast<cl_ulong>(indices.storage_offset()));
    kernel.setArg(argument++, buffer(out));
    kernel.setArg(argument++, static_cast<cl_ulong>(out.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(out.numel()));
    kernel.setArg(argument++, static_cast<cl_ulong>(product(self.sizes(), dim + 1, self.dim())));
    kernel.setArg(argument++, static_cast<cl_ulong>(self.size(dim)));
    kernel.setArg(argument++, static_cast<cl_ulong>(index.numel()));
    kernel.setArg(argument++, static_cast<cl_int>(0));
    kernel.setArg(argument++, buffer(error_flag));
    enqueue(self, kernel, out.numel(), "coeiroink_index_select");
    check_error_flag(self, error_flag, "index_select index out of range");
    return out;
}

Tensor flip(const Tensor& self, at::IntArrayRef dims) {
    TORCH_CHECK(self.dim() <= 8, "OpenCL flip supports at most 8 dimensions");
    uint64_t flip_mask = 0;
    for (int64_t dim : dims) {
        dim = normalize_dim(dim, self.dim());
        uint64_t bit = uint64_t{1} << dim;
        TORCH_CHECK((flip_mask & bit) == 0, "flip dimensions must be unique");
        flip_mask |= bit;
    }
    Tensor input = self.contiguous();
    Tensor out = new_tensor(shape(self.sizes()), self, self.scalar_type());
    if (out.numel() == 0) return out;

    const auto& compiled = program(
        self,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", opencl_type(self.scalar_type()))});
    cl::Kernel kernel(compiled, "flip_value");
    auto dimensions = padded_dimensions(self.sizes());
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(out));
    kernel.setArg(argument++, static_cast<cl_ulong>(out.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(out.numel()));
    kernel.setArg(argument++, static_cast<cl_int>(self.dim()));
    kernel.setArg(argument++, static_cast<cl_ulong>(flip_mask));
    for (cl_ulong dimension : dimensions) kernel.setArg(argument++, dimension);
    enqueue(self, kernel, out.numel(), "coeiroink_flip");
    return out;
}

Tensor index_tensor(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
    Tensor index = single_index(self, indices);
    if (index.scalar_type() == at::kLong) {
        // int64索引はPython互換の負数折返しを有効にしたgatherとして実行する。
        std::vector<int64_t> sizes = shape(index.sizes());
        sizes.insert(sizes.end(), self.sizes().begin() + 1, self.sizes().end());
        Tensor out = new_tensor(sizes, self, self.scalar_type());
        if (out.numel() == 0) return out;

        Tensor input = self.contiguous();
        Tensor contiguous_index = index.contiguous();
        Tensor error_flag = new_error_flag(self);
        int64_t inner = product(self.sizes(), 1, self.dim());
        const auto& compiled = program(
            self,
            "coeiroink_vits_index",
            {dlprim::gpu::Parameter("VALUE_TYPE", opencl_type(self.scalar_type()))});
        cl::Kernel kernel(compiled, "index_select_value");
        int argument = 0;
        kernel.setArg(argument++, buffer(input));
        kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
        kernel.setArg(argument++, buffer(contiguous_index));
        kernel.setArg(argument++, static_cast<cl_ulong>(contiguous_index.storage_offset()));
        kernel.setArg(argument++, buffer(out));
        kernel.setArg(argument++, static_cast<cl_ulong>(out.storage_offset()));
        kernel.setArg(argument++, static_cast<cl_ulong>(out.numel()));
        kernel.setArg(argument++, static_cast<cl_ulong>(inner));
        kernel.setArg(argument++, static_cast<cl_ulong>(self.size(0)));
        kernel.setArg(argument++, static_cast<cl_ulong>(index.numel()));
        kernel.setArg(argument++, static_cast<cl_int>(1));
        kernel.setArg(argument++, buffer(error_flag));
        enqueue(self, kernel, out.numel(), "coeiroink_index_tensor");
        check_error_flag(self, error_flag, "index out of range");
        return out;
    }

    // bool索引は選択数から出力形状を確定した後、選択された行をGPU上で詰める。
    TORCH_CHECK(index.scalar_type() == at::kBool, "OpenCL index must use bool or int64");
    TORCH_CHECK(index.dim() <= self.dim(), "boolean index has too many dimensions");
    for (int64_t axis = 0; axis < index.dim(); ++axis) {
        TORCH_CHECK(index.size(axis) == self.size(axis), "boolean index shape mismatch");
    }
    uint32_t selected_count = count_true_values(index);
    std::vector<int64_t> sizes{static_cast<int64_t>(selected_count)};
    sizes.insert(sizes.end(), self.sizes().begin() + index.dim(), self.sizes().end());
    Tensor out = new_tensor(sizes, self, self.scalar_type());
    if (out.numel() == 0) return out;

    Tensor input = self.contiguous();
    Tensor mask = index.contiguous();
    int64_t inner = product(self.sizes(), index.dim(), self.dim());
    const auto& compiled = program(
        self,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", opencl_type(self.scalar_type()))});
    cl::Kernel kernel(compiled, "masked_index_value");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(mask));
    kernel.setArg(argument++, static_cast<cl_ulong>(mask.storage_offset()));
    kernel.setArg(argument++, buffer(out));
    kernel.setArg(argument++, static_cast<cl_ulong>(out.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(mask.numel()));
    kernel.setArg(argument++, static_cast<cl_ulong>(inner));
    enqueue(self, kernel, 1, "coeiroink_masked_index");
    return out;
}

Tensor& index_tensor_out(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    Tensor& out) {
    copy_to_out(index_tensor(self, indices), out);
    return out;
}

Tensor& index_put_impl(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& values,
    bool accumulate,
    bool /*unsafe*/) {
    TORCH_CHECK(!accumulate, "OpenCL VITS index_put does not use accumulation");
    Tensor index = single_index(self, indices);
    TORCH_CHECK(index.scalar_type() == at::kBool, "OpenCL VITS index_put requires a bool mask");
    TORCH_CHECK(index.dim() <= self.dim(), "boolean index has too many dimensions");
    for (int64_t axis = 0; axis < index.dim(); ++axis) {
        TORCH_CHECK(index.size(axis) == self.size(axis), "boolean index shape mismatch");
    }

    uint32_t selected_count = count_true_values(index);
    int64_t inner = product(self.sizes(), index.dim(), self.dim());
    int64_t expected = static_cast<int64_t>(selected_count) * inner;
    Tensor prepared_values = contiguous_on_device(values, self, self.scalar_type());
    TORCH_CHECK(
        prepared_values.numel() == 1 || prepared_values.numel() == expected,
        "index_put values do not match the selected shape");
    if (selected_count == 0) return self;

    Tensor target = self.contiguous();
    Tensor mask = index.contiguous();
    const auto& compiled = program(
        self,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", opencl_type(self.scalar_type()))});
    cl::Kernel kernel(compiled, "masked_index_put_value");
    int argument = 0;
    kernel.setArg(argument++, buffer(target));
    kernel.setArg(argument++, static_cast<cl_ulong>(target.storage_offset()));
    kernel.setArg(argument++, buffer(mask));
    kernel.setArg(argument++, static_cast<cl_ulong>(mask.storage_offset()));
    kernel.setArg(argument++, buffer(prepared_values));
    kernel.setArg(argument++, static_cast<cl_ulong>(prepared_values.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(prepared_values.numel()));
    kernel.setArg(argument++, static_cast<cl_ulong>(mask.numel()));
    kernel.setArg(argument++, static_cast<cl_ulong>(inner));
    enqueue(self, kernel, 1, "coeiroink_masked_index_put");
    if (!self.is_contiguous()) self.copy_(target);
    return self;
}

Tensor& index_put(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& values,
    bool accumulate) {
    return index_put_impl(self, indices, values, accumulate, false);
}

Tensor nonzero(const Tensor& self) {
    TORCH_CHECK(self.scalar_type() == at::kBool, "OpenCL VITS nonzero requires a bool tensor");
    TORCH_CHECK(self.dim() <= 8, "OpenCL nonzero supports at most 8 dimensions");
    uint32_t count = count_true_values(self);
    Tensor out = new_tensor({static_cast<int64_t>(count), self.dim()}, self, at::kLong);
    if (count == 0 || self.dim() == 0) return out;

    Tensor input = self.contiguous();
    const auto& compiled = program(
        self,
        "coeiroink_vits_index",
        {dlprim::gpu::Parameter("VALUE_TYPE", "uchar")});
    cl::Kernel kernel(compiled, "nonzero_bool");
    auto dimensions = padded_dimensions(self.sizes());
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(out));
    kernel.setArg(argument++, static_cast<cl_ulong>(out.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.numel()));
    kernel.setArg(argument++, static_cast<cl_int>(self.dim()));
    for (cl_ulong dimension : dimensions) kernel.setArg(argument++, dimension);
    enqueue(self, kernel, 1, "coeiroink_nonzero");
    return out;
}

Tensor& nonzero_out(const Tensor& self, Tensor& out) {
    copy_to_out(coeiroink::opencl::nonzero(self), out);
    return out;
}

Tensor& cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    Tensor& out) {
    // 対応範囲はVITSが要求するfloat32・最終軸に限定し、別形式は明示的に失敗させる。
    dim = normalize_dim(dim, self.dim());
    TORCH_CHECK(dim == self.dim() - 1, "OpenCL VITS cumsum supports the last dimension");
    TORCH_CHECK(self.scalar_type() == at::kFloat, "OpenCL VITS cumsum requires float32");
    TORCH_CHECK(dtype.value_or(at::kFloat) == at::kFloat, "OpenCL VITS cumsum output must be float32");
    if (!out.sizes().equals(self.sizes())) out.resize_(self.sizes());
    TORCH_CHECK(out.scalar_type() == at::kFloat, "cumsum output dtype mismatch");
    if (self.numel() == 0) return out;

    Tensor input = self.contiguous();
    Tensor output = out.contiguous();
    int64_t width = self.size(dim);
    int64_t lines = self.numel() / width;
    const auto& compiled = program(self, "coeiroink_vits_reduction");
    cl::Kernel kernel(compiled, "cumsum_last_float");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(output));
    kernel.setArg(argument++, static_cast<cl_ulong>(output.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(lines));
    kernel.setArg(argument++, static_cast<cl_ulong>(width));
    enqueue(self, kernel, lines, "coeiroink_cumsum");
    if (!out.is_contiguous()) out.copy_(output);
    return out;
}

Tensor cumsum(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
    Tensor out = new_tensor(shape(self.sizes()), self, dtype.value_or(self.scalar_type()));
    cumsum_out(self, dim, dtype, out);
    return out;
}

Tensor& softplus_out(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    Tensor& out) {
    TORCH_CHECK(self.scalar_type() == at::kFloat, "OpenCL VITS softplus requires float32");
    if (!out.sizes().equals(self.sizes())) out.resize_(self.sizes());
    TORCH_CHECK(out.scalar_type() == at::kFloat, "softplus output dtype mismatch");
    if (self.numel() == 0) return out;

    Tensor input = self.contiguous();
    Tensor output = out.contiguous();
    dlprim::core::pointwise_operation(
        {ptdlprim::todp(input)},
        {ptdlprim::todp(output)},
        {beta.toDouble(), threshold.toDouble()},
        "dtype z=x0*w0; y0=z>w1 ? x0 : (max(z,(dtype)0)+log1p(exp(-fabs(z))))/w0;",
        ptdlprim::getExecutionContext(self));
    if (!out.is_contiguous()) out.copy_(output);
    return out;
}

Tensor softplus(const Tensor& self, const Scalar& beta, const Scalar& threshold) {
    Tensor out = new_tensor(shape(self.sizes()), self, self.scalar_type());
    softplus_out(self, beta, threshold, out);
    return out;
}

Tensor relu(const Tensor& self) {
    Tensor input = self.contiguous();
    Tensor out = new_tensor(shape(self.sizes()), self, self.scalar_type());
    if (self.numel() == 0) return out;

    dlprim::Tensor input_view = ptdlprim::todp(input);
    dlprim::Tensor output_view = ptdlprim::todp(out);
    dlprim::core::activation_forward(
        input_view,
        output_view,
        dlprim::StandardActivations::relu,
        ptdlprim::getExecutionContext(self));
    return out;
}

Tensor& masked_fill(Tensor& self, const Tensor& mask, const Scalar& value) {
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill mask must use bool");
    TORCH_CHECK(mask.device() == self.device(), "masked_fill mask must use the input device");
    Tensor target = self.contiguous();
    Tensor contiguous_mask = mask.contiguous();
    dlprim::core::pointwise_operation_broadcast(
        {ptdlprim::todp(target), ptdlprim::todp(contiguous_mask)},
        {ptdlprim::todp(target)},
        {value.toDouble()},
        "y0 = x1 ? w0 : x0;",
        ptdlprim::getExecutionContext(self));
    if (!self.is_contiguous()) self.copy_(target);
    return self;
}

Tensor& clamp_tensor_out(
    const Tensor& self,
    const c10::optional<Tensor>& minimum,
    const c10::optional<Tensor>& maximum,
    Tensor& out) {
    if (!out.sizes().equals(self.sizes())) out.resize_(self.sizes());
    TORCH_CHECK(out.scalar_type() == self.scalar_type(), "clamp output dtype mismatch");
    auto check_bound = [&](const c10::optional<Tensor>& bound) {
        if (!bound) return;
        TORCH_CHECK(bound->device() == self.device(), "clamp bounds must use the input device");
        TORCH_CHECK(
            bound->scalar_type() == self.scalar_type(),
            "OpenCL VITS clamp requires bounds with the input dtype");
    };
    check_bound(minimum);
    check_bound(maximum);
    Tensor input = self.contiguous();
    Tensor output = out.contiguous();
    if (!minimum && !maximum) {
        output.copy_(input);
    } else if (minimum && maximum) {
        Tensor min_value = contiguous_on_device(*minimum, self, self.scalar_type());
        Tensor max_value = contiguous_on_device(*maximum, self, self.scalar_type());
        dlprim::core::pointwise_operation_broadcast(
            {ptdlprim::todp(input), ptdlprim::todp(min_value), ptdlprim::todp(max_value)},
            {ptdlprim::todp(output)},
            {},
            "y0 = min(max(x0,x1),x2);",
            ptdlprim::getExecutionContext(self));
    } else {
        Tensor bound = contiguous_on_device(minimum ? *minimum : *maximum, self, self.scalar_type());
        dlprim::core::pointwise_operation_broadcast(
            {ptdlprim::todp(input), ptdlprim::todp(bound)},
            {ptdlprim::todp(output)},
            {},
            minimum ? "y0 = max(x0,x1);" : "y0 = min(x0,x1);",
            ptdlprim::getExecutionContext(self));
    }
    if (!out.is_contiguous()) out.copy_(output);
    return out;
}

Tensor clamp_tensor(
    const Tensor& self,
    const c10::optional<Tensor>& minimum,
    const c10::optional<Tensor>& maximum) {
    Tensor out = new_tensor(shape(self.sizes()), self, self.scalar_type());
    clamp_tensor_out(self, minimum, maximum, out);
    return out;
}

std::tuple<Tensor&, Tensor&> max_dim_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
    dim = normalize_dim(dim, self.dim());
    TORCH_CHECK(dim == self.dim() - 1, "OpenCL VITS max supports the last dimension");
    TORCH_CHECK(self.scalar_type() == at::kFloat, "OpenCL VITS max requires float32");
    TORCH_CHECK(self.size(dim) > 0, "max reduction dimension must be non-empty");
    std::vector<int64_t> sizes = shape(self.sizes());
    if (keepdim) {
        sizes[dim] = 1;
    } else {
        sizes.erase(sizes.begin() + dim);
    }
    if (!values.sizes().equals(sizes)) values.resize_(sizes);
    if (!indices.sizes().equals(sizes)) indices.resize_(sizes);
    TORCH_CHECK(values.scalar_type() == at::kFloat, "max values must use float32");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "max indices must use int64");
    at::assert_no_internal_overlap(values);
    at::assert_no_internal_overlap(indices);
    at::assert_no_overlap(values, self);
    at::assert_no_overlap(indices, self);
    at::assert_no_overlap(values, indices);

    Tensor input = self.contiguous();
    Tensor output_values = values.contiguous();
    Tensor output_indices = indices.contiguous();
    int64_t lines = self.numel() / self.size(dim);
    const auto& compiled = program(self, "coeiroink_vits_reduction");
    cl::Kernel kernel(compiled, "max_last_float");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(output_values));
    kernel.setArg(argument++, static_cast<cl_ulong>(output_values.storage_offset()));
    kernel.setArg(argument++, buffer(output_indices));
    kernel.setArg(argument++, static_cast<cl_ulong>(output_indices.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(lines));
    kernel.setArg(argument++, static_cast<cl_ulong>(self.size(dim)));
    enqueue(self, kernel, lines, "coeiroink_max");
    if (!values.is_contiguous()) values.copy_(output_values);
    if (!indices.is_contiguous()) indices.copy_(output_indices);
    return {values, indices};
}

std::tuple<Tensor, Tensor> max_dim(const Tensor& self, int64_t dim, bool keepdim) {
    int64_t normalized = normalize_dim(dim, self.dim());
    std::vector<int64_t> sizes = shape(self.sizes());
    if (keepdim) {
        sizes[normalized] = 1;
    } else {
        sizes.erase(sizes.begin() + normalized);
    }
    Tensor values = new_tensor(sizes, self, self.scalar_type());
    Tensor indices = new_tensor(sizes, self, at::kLong);
    max_dim_out(self, dim, keepdim, values, indices);
    return {values, indices};
}

Tensor& all_out(const Tensor& self, Tensor& out) {
    TORCH_CHECK(self.scalar_type() == at::kBool, "OpenCL VITS all requires bool");
    if (out.dim() != 0) out.resize_({});
    TORCH_CHECK(out.scalar_type() == at::kBool, "all output must use bool");
    Tensor input = self.contiguous();
    Tensor output = out.contiguous();
    const auto& compiled = program(self, "coeiroink_vits_reduction");
    cl::Kernel kernel(compiled, "all_bool");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(output));
    kernel.setArg(argument++, static_cast<cl_ulong>(output.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.numel()));
    enqueue(self, kernel, 1, "coeiroink_all");
    if (!out.is_contiguous()) out.copy_(output);
    return out;
}

Tensor all(const Tensor& self) {
    Tensor out = new_tensor({}, self, at::kBool);
    all_out(self, out);
    return out;
}

std::tuple<Tensor, Tensor> weight_norm(const Tensor& v, const Tensor& g, int64_t dim) {
    dim = normalize_dim(dim, v.dim());
    TORCH_CHECK(dim == 0, "OpenCL VITS weight_norm supports dim=0");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "weight_norm v must use float32");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "weight_norm g must use float32");
    TORCH_CHECK(g.device() == v.device(), "weight_norm tensors must use the same device");
    int64_t groups = v.size(0);
    TORCH_CHECK(g.numel() == groups, "weight_norm g shape does not match v");

    Tensor input = v.contiguous();
    Tensor scale = g.contiguous();
    Tensor weight = new_tensor(shape(v.sizes()), v, at::kFloat);
    Tensor norms = new_tensor(shape(g.sizes()), v, at::kFloat);
    const auto& compiled = program(v, "coeiroink_vits_reduction");
    cl::Kernel kernel(compiled, "weight_norm_dim0_float");
    int argument = 0;
    kernel.setArg(argument++, buffer(input));
    kernel.setArg(argument++, static_cast<cl_ulong>(input.storage_offset()));
    kernel.setArg(argument++, buffer(scale));
    kernel.setArg(argument++, static_cast<cl_ulong>(scale.storage_offset()));
    kernel.setArg(argument++, buffer(weight));
    kernel.setArg(argument++, static_cast<cl_ulong>(weight.storage_offset()));
    kernel.setArg(argument++, buffer(norms));
    kernel.setArg(argument++, static_cast<cl_ulong>(norms.storage_offset()));
    kernel.setArg(argument++, static_cast<cl_ulong>(groups));
    kernel.setArg(argument++, static_cast<cl_ulong>(v.numel() / groups));
    enqueue(v, kernel, groups, "coeiroink_weight_norm");
    return {weight, norms};
}

std::tuple<Tensor&, Tensor&> weight_norm_out(
    const Tensor& v,
    const Tensor& g,
    int64_t dim,
    Tensor& weight,
    Tensor& norms) {
    auto [computed_weight, computed_norms] = weight_norm(v, g, dim);
    copy_to_out(computed_weight, weight);
    copy_to_out(computed_norms, norms);
    return {weight, norms};
}

}  // namespace coeiroink::opencl

// pytorch_dlprimに不足するVITS演算だけをPrivateUse1へ追加登録する。
// 未登録の演算はpytorch_dlprim側の警告とフォールバックに任せ、ここでは隠さない。
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::gather", &coeiroink::opencl::gather);
    m.impl("aten::gather.out", &coeiroink::opencl::gather_out);
    m.impl("aten::index_select", &coeiroink::opencl::index_select);
    m.impl("aten::flip", &coeiroink::opencl::flip);
    m.impl("aten::index.Tensor", &coeiroink::opencl::index_tensor);
    m.impl("aten::index.Tensor_out", &coeiroink::opencl::index_tensor_out);
    m.impl("aten::index_put_", &coeiroink::opencl::index_put);
    m.impl("aten::_index_put_impl_", &coeiroink::opencl::index_put_impl);
    m.impl("aten::nonzero", &coeiroink::opencl::nonzero);
    m.impl("aten::nonzero.out", &coeiroink::opencl::nonzero_out);
    m.impl("aten::cumsum", &coeiroink::opencl::cumsum);
    m.impl("aten::cumsum.out", &coeiroink::opencl::cumsum_out);
    m.impl("aten::softplus", &coeiroink::opencl::softplus);
    m.impl("aten::softplus.out", &coeiroink::opencl::softplus_out);
    m.impl("aten::relu", &coeiroink::opencl::relu);
    m.impl("aten::masked_fill_.Scalar", &coeiroink::opencl::masked_fill);
    m.impl("aten::clamp.Tensor", &coeiroink::opencl::clamp_tensor);
    m.impl("aten::clamp.Tensor_out", &coeiroink::opencl::clamp_tensor_out);
    m.impl("aten::max.dim", &coeiroink::opencl::max_dim);
    m.impl("aten::max.dim_max", &coeiroink::opencl::max_dim_out);
    m.impl("aten::all", &coeiroink::opencl::all);
    m.impl("aten::all.all_out", &coeiroink::opencl::all_out);
    m.impl("aten::_weight_norm_interface", &coeiroink::opencl::weight_norm);
    m.impl("aten::_weight_norm_interface.out", &coeiroink::opencl::weight_norm_out);
}
