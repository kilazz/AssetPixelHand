// =======================================================================================
//  decoder.cpp - High-Performance C++ Python Module for DDS File Decoding
// =======================================================================================
// Provides Python bindings for the DirectXTex library. This version relies on the
// modern capabilities of the library to handle complex, legacy, and console formats,
// removing the need for manual fallback logic.
// =======================================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>

#define NOMINMAX
#include <Windows.h>
#include <objbase.h>
#include "DirectXTex.h"

namespace py = pybind11;
using namespace DirectX;

// HRESULT codes - kept for COM initialization
constexpr HRESULT CUSTOM_RPC_E_CHANGED_MODE  = static_cast<HRESULT>(0x80010106);

// =======================================================================================
// Utility Functions
// =======================================================================================

// Forward declaration
std::string DXGIFormatToString(DXGI_FORMAT format);

class CoInitializer {
public:
    CoInitializer() {
        hr_ = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (hr_ == CUSTOM_RPC_E_CHANGED_MODE) {
            hr_ = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        }
        should_uninit_ = SUCCEEDED(hr_);
        if (hr_ == S_FALSE) {
            hr_ = S_OK;
            should_uninit_ = false;
        }
    }
    ~CoInitializer() {
        if (should_uninit_) {
            CoUninitialize();
        }
    }
    bool IsValid() const { return SUCCEEDED(hr_); }
    HRESULT GetResult() const { return hr_; }

private:
    HRESULT hr_;
    bool should_uninit_ = true;
};

std::string HResultToString(HRESULT hr) {
    char* msg_buf = nullptr;
    DWORD result = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&msg_buf), 0, nullptr
    );

    std::string msg;
    if (result > 0 && msg_buf) {
        msg = msg_buf;
        LocalFree(msg_buf);
        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n' || msg.back() == ' ')) {
            msg.pop_back();
        }
    } else {
        std::ostringstream oss;
        oss << "HRESULT 0x" << std::hex << std::uppercase << hr;
        msg = oss.str();
    }
    return msg;
}

// Manual conversion function for 16-bit float (half) to 32-bit float
float half_to_float(uint16_t half) {
    uint32_t sign = (half >> 15) & 0x0001;
    uint32_t exponent = (half >> 10) & 0x001F;
    uint32_t mantissa = half & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) { // Zero
            return sign ? -0.0f : 0.0f;
        } else { // Subnormal
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x0400;
        }
    } else if (exponent == 31) {
        return mantissa == 0
            ? (sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity()) // Infinity
            : std::numeric_limits<float>::quiet_NaN(); // NaN
    }

    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;

    uint32_t result_bits = (sign << 31) | (exponent << 23) | mantissa;
    float result;
    memcpy(&result, &result_bits, sizeof(result));
    return result;
}

// =======================================================================================
// NumPy Array Creation Helper
// =======================================================================================

py::object CreateNumpyArrayFromImage(const DirectX::Image* image) {
    if (!image || !image->pixels) {
        throw std::runtime_error("Cannot create NumPy array from an invalid image.");
    }

    // Helper for direct memory copy into a new NumPy array
    auto direct_copy = [&](auto dtype_example, int channels) -> py::object {
        using T = decltype(dtype_example);
        std::vector<py::ssize_t> shape = {(py::ssize_t)image->height, (py::ssize_t)image->width};
        if (channels > 0) {
            shape.push_back(channels);
        }

        auto arr = py::array_t<T>(shape);
        py::buffer_info buf = arr.request();
        size_t numpy_row_pitch = image->width * sizeof(T) * (channels > 0 ? channels : 1);

        if (image->rowPitch == numpy_row_pitch) {
            memcpy(buf.ptr, image->pixels, image->slicePitch);
        } else {
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                memcpy(dst_ptr + y * numpy_row_pitch, src_ptr + y * image->rowPitch, numpy_row_pitch);
            }
        }
        return arr;
    };

    // Helper to convert 16-bit float (half) to 32-bit float
    auto convert_half_to_float = [&](size_t num_components) {
        std::vector<py::ssize_t> shape = {(py::ssize_t)image->height, (py::ssize_t)image->width};
        if (num_components > 1) {
            shape.push_back(num_components);
        }
        auto numpy_array = py::array_t<float>(shape);
        py::buffer_info buf = numpy_array.request();
        auto* dst_ptr = static_cast<float*>(buf.ptr);
        const auto* src_ptr_base = image->pixels;

        for (size_t y = 0; y < image->height; ++y) {
            const auto* src_row_ptr = reinterpret_cast<const uint16_t*>(src_ptr_base + y * image->rowPitch);
            float* dst_row_ptr = dst_ptr + y * image->width * num_components;
            for (size_t i = 0; i < image->width * num_components; ++i) {
                dst_row_ptr[i] = half_to_float(src_row_ptr[i]);
            }
        }
        return py::object(numpy_array);
    };

    // The core logic: decide what to return based on the image format.
    switch (image->format) {
        // --- 32-bit Floating Point (HDR) -> float32 array ---
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return direct_copy(float(), 4);
        case DXGI_FORMAT_R32G32B32_FLOAT:    return direct_copy(float(), 3);
        case DXGI_FORMAT_R32G32_FLOAT:       return direct_copy(float(), 2);
        case DXGI_FORMAT_R32_FLOAT:          return direct_copy(float(), 1);

        // --- 16-bit Floating Point (Half/HDR) -> float32 array ---
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return convert_half_to_float(4);
        case DXGI_FORMAT_R16G16_FLOAT:       return convert_half_to_float(2);
        case DXGI_FORMAT_R16_FLOAT:          return convert_half_to_float(1);

        // --- 16-bit Integer -> uint16/int16 array ---
        case DXGI_FORMAT_R16G16B16A16_UNORM: return direct_copy(uint16_t(), 4);
        case DXGI_FORMAT_R16G16_UNORM:       return direct_copy(uint16_t(), 2);
        case DXGI_FORMAT_R16_UNORM:          return direct_copy(uint16_t(), 1);
        case DXGI_FORMAT_R16G16B16A16_UINT:  return direct_copy(uint16_t(), 4);
        case DXGI_FORMAT_R16G16_UINT:        return direct_copy(uint16_t(), 2);
        case DXGI_FORMAT_R16_UINT:           return direct_copy(uint16_t(), 1);
        case DXGI_FORMAT_R16G16B16A16_SINT:  return direct_copy(int16_t(), 4);
        case DXGI_FORMAT_R16G16_SINT:        return direct_copy(int16_t(), 2);
        case DXGI_FORMAT_R16_SINT:           return direct_copy(int16_t(), 1);

        // --- 8-bit UNORM (SDR) -> uint8 array ---
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return direct_copy(uint8_t(), 4);

        // BGRA format -> requires channel swap to RGBA, returns uint8 array
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    dst_row[x * 4 + 0] = src_row[x * 4 + 2]; // R
                    dst_row[x * 4 + 1] = src_row[x * 4 + 1]; // G
                    dst_row[x * 4 + 2] = src_row[x * 4 + 0]; // B
                    dst_row[x * 4 + 3] = src_row[x * 4 + 3]; // A
                }
            }
            return arr;
        }

        // --- 8-bit Integer -> uint8/int8 array ---
        case DXGI_FORMAT_R8G8B8A8_UINT: return direct_copy(uint8_t(), 4);
        case DXGI_FORMAT_R8G8_UINT:     return direct_copy(uint8_t(), 2);
        case DXGI_FORMAT_R8_UINT:       return direct_copy(uint8_t(), 1);
        case DXGI_FORMAT_R8G8B8A8_SINT: return direct_copy(int8_t(), 4);
        case DXGI_FORMAT_R8G8_SINT:     return direct_copy(int8_t(), 2);
        case DXGI_FORMAT_R8_SINT:       return direct_copy(int8_t(), 1);

        // --- SNORM formats -> convert to float32 to preserve [-1, 1] range ---
        case DXGI_FORMAT_R8G8B8A8_SNORM:
        case DXGI_FORMAT_R8G8_SNORM:
        case DXGI_FORMAT_R8_SNORM:
        case DXGI_FORMAT_R16G16B16A16_SNORM:
        case DXGI_FORMAT_R16G16_SNORM:
        case DXGI_FORMAT_R16_SNORM: {
            int channels;
            bool is16bit = false;
            if (image->format == DXGI_FORMAT_R8G8B8A8_SNORM || image->format == DXGI_FORMAT_R16G16B16A16_SNORM) channels = 4;
            else if (image->format == DXGI_FORMAT_R8G8_SNORM || image->format == DXGI_FORMAT_R16G16_SNORM) channels = 2;
            else channels = 1;

            if (image->format == DXGI_FORMAT_R16G16B16A16_SNORM || image->format == DXGI_FORMAT_R16G16_SNORM || image->format == DXGI_FORMAT_R16_SNORM) is16bit = true;

            auto arr = py::array_t<float>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, (py::ssize_t)channels});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<float*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const void* src_row = src_ptr + y * image->rowPitch;
                float* dst_row = dst_ptr + y * image->width * channels;
                for (size_t x = 0; x < image->width * channels; ++x) {
                    if (is16bit) {
                        dst_row[x] = std::max(-1.f, static_cast<float>(reinterpret_cast<const int16_t*>(src_row)[x]) / 32767.f);
                    } else {
                        dst_row[x] = std::max(-1.f, static_cast<float>(reinterpret_cast<const int8_t*>(src_row)[x]) / 127.f);
                    }
                }
            }
            return arr;
        }

        // --- Packed 16-bit formats -> unpack to RGBA8 uint8 array ---
        case DXGI_FORMAT_B5G6R5_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const auto* src_row = reinterpret_cast<const uint16_t*>(src_ptr + y * image->rowPitch);
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint16_t p = src_row[x];
                    dst_row[x * 4 + 0] = static_cast<uint8_t>(((p >> 11) & 0x1F) * 255 / 31); // R
                    dst_row[x * 4 + 1] = static_cast<uint8_t>(((p >> 5)  & 0x3F) * 255 / 63); // G
                    dst_row[x * 4 + 2] = static_cast<uint8_t>(( p        & 0x1F) * 255 / 31); // B
                    dst_row[x * 4 + 3] = 255;                                                 // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_B5G5R5A1_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const auto* src_row = reinterpret_cast<const uint16_t*>(src_ptr + y * image->rowPitch);
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint16_t p = src_row[x];
                    dst_row[x * 4 + 0] = static_cast<uint8_t>(((p >> 10) & 0x1F) * 255 / 31); // R
                    dst_row[x * 4 + 1] = static_cast<uint8_t>(((p >> 5)  & 0x1F) * 255 / 31); // G
                    dst_row[x * 4 + 2] = static_cast<uint8_t>(( p        & 0x1F) * 255 / 31); // B
                    dst_row[x * 4 + 3] = static_cast<uint8_t>(((p >> 15) & 0x01) * 255);      // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_B4G4R4A4_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const auto* src_row = reinterpret_cast<const uint16_t*>(src_ptr + y * image->rowPitch);
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint16_t p = src_row[x];
                    uint8_t r = (p >> 8) & 0x0F, g = (p >> 4) & 0x0F, b = p & 0x0F, a = (p >> 12) & 0x0F;
                    dst_row[x * 4 + 0] = (r << 4) | r; // R
                    dst_row[x * 4 + 1] = (g << 4) | g; // G
                    dst_row[x * 4 + 2] = (b << 4) | b; // B
                    dst_row[x * 4 + 3] = (a << 4) | a; // A
                }
            }
            return arr;
        }

        // --- Single and Dual channel UNORM formats -> expand to RGBA8 for visualization ---
        case DXGI_FORMAT_R8G8_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    dst_row[x * 4 + 0] = src_row[x * 2 + 0]; // R
                    dst_row[x * 4 + 1] = src_row[x * 2 + 1]; // G
                    dst_row[x * 4 + 2] = 0;                  // B
                    dst_row[x * 4 + 3] = 255;                // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_R8_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint8_t val = src_row[x];
                    dst_row[x * 4 + 0] = val; // R
                    dst_row[x * 4 + 1] = val; // G
                    dst_row[x * 4 + 2] = val; // B
                    dst_row[x * 4 + 3] = 255; // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_A8_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            auto* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const auto* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    dst_row[x * 4 + 0] = 255;        // R
                    dst_row[x * 4 + 1] = 255;        // G
                    dst_row[x * 4 + 2] = 255;        // B
                    dst_row[x * 4 + 3] = src_row[x]; // A
                }
            }
            return arr;
        }

        default:
            throw std::runtime_error("Unsupported DXGI_FORMAT for NumPy conversion: " + DXGIFormatToString(image->format));
    }
}

// =======================================================================================
// Main DDS Decoding Function
// =======================================================================================

py::dict decode_dds(const py::bytes& dds_bytes, size_t mip_level = 0, py::ssize_t array_index = -1, bool force_rgba8 = false) {
    py::buffer_info info(py::buffer(dds_bytes).request());
    // Basic validation
    if (info.size < 4) {
        throw py::value_error("Input data too small for a DDS file");
    }
    if (memcmp(info.ptr, "DDS ", 4) != 0) {
        throw py::value_error("Not a DDS file (missing 'DDS ' magic number)");
    }

    py::gil_scoped_release gil_release;
    CoInitializer com_init;
    if (!com_init.IsValid()) {
        py::gil_scoped_acquire gil_acquire;
        throw std::runtime_error("COM initialization failed: " + HResultToString(com_init.GetResult()));
    }

    const auto* dds_data_ptr = static_cast<const uint8_t*>(info.ptr);
    const size_t dds_data_size = static_cast<size_t>(info.size);
    auto image = std::make_unique<ScratchImage>();

    // Use permissive flags to allow DirectXTex to handle legacy/console formats automatically.
    DDS_FLAGS flags = DDS_FLAGS_PERMISSIVE | DDS_FLAGS_ALLOW_LARGE_FILES;
    HRESULT hr = LoadFromDDSMemory(dds_data_ptr, dds_data_size, flags, nullptr, *image);

    if (FAILED(hr)) {
        // If the modern loader fails, there's no need for a manual fallback.
        // The library is now robust enough to be the single source of truth.
        py::gil_scoped_acquire gil_acquire;
        throw std::runtime_error("DirectXTex failed to load DDS memory: " + HResultToString(hr));
    }

    const auto& metadata = image->GetMetadata();
    if (mip_level >= metadata.mipLevels) {
        py::gil_scoped_acquire gil_acquire;
        throw py::value_error("Mipmap level is out of bounds");
    }

    auto process_single_image = [&](const DirectX::Image* input_image) -> std::unique_ptr<ScratchImage> {
        if (!input_image) throw std::runtime_error("Input image for processing is null");

        auto current_image = std::make_unique<ScratchImage>();
        HRESULT hr_process = current_image->InitializeFromImage(*input_image);
        if (FAILED(hr_process)) throw std::runtime_error("Failed to initialize from selected image: " + HResultToString(hr_process));

        if (DirectX::IsCompressed(current_image->GetMetadata().format)) {
            auto temp = std::make_unique<ScratchImage>();
            hr_process = Decompress(*current_image->GetImage(0, 0, 0), DXGI_FORMAT_UNKNOWN, *temp);
            if (FAILED(hr_process)) throw std::runtime_error("Decompression failed: " + HResultToString(hr_process));
            current_image = std::move(temp);
        }

        if (force_rgba8) {
            DXGI_FORMAT current_format = current_image->GetMetadata().format;
            if (current_format != DXGI_FORMAT_R8G8B8A8_UNORM && current_format != DXGI_FORMAT_B8G8R8A8_UNORM) {
                auto temp = std::make_unique<ScratchImage>();
                hr_process = Convert(*current_image->GetImage(0, 0, 0), DXGI_FORMAT_R8G8B8A8_UNORM, TEX_FILTER_DEFAULT, TEX_THRESHOLD_DEFAULT, *temp);
                if (FAILED(hr_process)) throw std::runtime_error("Format conversion to RGBA8 failed: " + HResultToString(hr_process));
                current_image = std::move(temp);
            }
        }
        return current_image;
    };

    bool load_all = array_index < 0;
    bool is_3d = metadata.dimension == TEX_DIMENSION_TEXTURE3D;
    size_t num_images_to_process = load_all ? (is_3d ? metadata.depth : metadata.arraySize) : 1;

    if (!load_all && array_index >= static_cast<py::ssize_t>(metadata.arraySize)) {
        py::gil_scoped_acquire gil_acquire;
        throw py::value_error("Array index is out of bounds");
    }

    std::vector<std::unique_ptr<ScratchImage>> processed_images;
    for (size_t i = 0; i < num_images_to_process; ++i) {
        size_t current_index = load_all ? i : static_cast<size_t>(array_index);
        const DirectX::Image* selected_image = image->GetImage(mip_level, current_index, 0);
        if (!selected_image) {
            throw std::runtime_error("Failed to get image slice for index " + std::to_string(current_index));
        }
        processed_images.push_back(process_single_image(selected_image));
    }

    const DirectX::Image* first_final_image = processed_images[0]->GetImage(0, 0, 0);
    if (!first_final_image || !first_final_image->pixels) {
        py::gil_scoped_acquire gil_acquire;
        throw std::runtime_error("Could not retrieve final processed image pixels");
    }

    py::gil_scoped_acquire gil_acquire;
    py::object numpy_array;

    if (num_images_to_process > 1) {
        py::array single_slice_template = py::cast<py::array>(CreateNumpyArrayFromImage(first_final_image));
        py::buffer_info slice_info = single_slice_template.request();

        std::vector<py::ssize_t> final_shape;
        final_shape.push_back(num_images_to_process);
        final_shape.insert(final_shape.end(), slice_info.shape.begin(), slice_info.shape.end());

        numpy_array = py::array(py::dtype(slice_info.format), final_shape);
        py::buffer_info final_buf = py::cast<py::array>(numpy_array).request();
        auto* final_ptr = static_cast<uint8_t*>(final_buf.ptr);

        size_t slice_size_bytes = 1;
        for (size_t dim = 1; dim < final_buf.ndim; ++dim) {
            slice_size_bytes *= final_buf.shape[dim];
        }
        slice_size_bytes *= final_buf.itemsize;

        for (size_t i = 0; i < num_images_to_process; ++i) {
            const DirectX::Image* current_slice = processed_images[i]->GetImage(0, 0, 0);
            uint8_t* dst_slice_ptr = final_ptr + i * slice_size_bytes;
            const uint8_t* src_slice_ptr = current_slice->pixels;

            size_t row_size_bytes = std::min(current_slice->rowPitch, current_slice->width * DirectX::BitsPerPixel(current_slice->format) / 8);

            for (size_t y = 0; y < current_slice->height; ++y) {
                 memcpy(dst_slice_ptr + y * row_size_bytes, src_slice_ptr + y * current_slice->rowPitch, row_size_bytes);
            }
        }
    } else {
        numpy_array = CreateNumpyArrayFromImage(first_final_image);
    }

    return py::dict(
        py::arg("width") = first_final_image->width,
        py::arg("height") = first_final_image->height,
        py::arg("depth") = (is_3d ? metadata.depth : 1),
        py::arg("array_size") = (!is_3d ? metadata.arraySize : 1),
        py::arg("data") = numpy_array,
        py::arg("format_str") = DXGIFormatToString(first_final_image->format),
        py::arg("mip_levels") = metadata.mipLevels,
        py::arg("is_cubemap") = metadata.IsCubemap()
    );
}

py::dict get_dds_metadata(const py::bytes& dds_bytes) {
    py::buffer_info info(py::buffer(dds_bytes).request());
    if (info.size < 4) {
        throw py::value_error("Input data too small for DDS header.");
    }
    if (memcmp(info.ptr, "DDS ", 4) != 0) {
        throw py::value_error("Input is not a valid DDS file.");
    }

    const auto* dds_data_ptr = static_cast<const uint8_t*>(info.ptr);
    const size_t dds_data_size = static_cast<size_t>(info.size);

    TexMetadata metadata;
    {
        py::gil_scoped_release release;
        // Use permissive flags to allow DirectXTex to handle legacy/console formats automatically.
        DDS_FLAGS flags = DDS_FLAGS_PERMISSIVE | DDS_FLAGS_ALLOW_LARGE_FILES;
        HRESULT hr = GetMetadataFromDDSMemory(dds_data_ptr, dds_data_size, flags, metadata);

        if (FAILED(hr)) {
            py::gil_scoped_acquire acquire;
            throw std::runtime_error("Failed to get metadata from DDS memory: " + HResultToString(hr));
        }
    }

    py::gil_scoped_acquire acquire;
    return py::dict(
        py::arg("width") = metadata.width,
        py::arg("height") = metadata.height,
        py::arg("depth") = metadata.depth,
        py::arg("format_str") = DXGIFormatToString(metadata.format),
        py::arg("mip_levels") = metadata.mipLevels,
        py::arg("array_size") = metadata.arraySize,
        py::arg("is_cubemap") = metadata.IsCubemap(),
        py::arg("is_3d") = (metadata.dimension == TEX_DIMENSION_TEXTURE3D)
    );
}

// =======================================================================================
// DXGI Format to String Conversion
// =======================================================================================
std::string DXGIFormatToString(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_UNKNOWN: return "UNKNOWN";
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: return "R32G32B32A32_TYPELESS";
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_UINT: return "R32G32B32A32_UINT";
        case DXGI_FORMAT_R32G32B32A32_SINT: return "R32G32B32A32_SINT";
        case DXGI_FORMAT_R32G32B32_TYPELESS: return "R32G32B32_TYPELESS";
        case DXGI_FORMAT_R32G32B32_FLOAT: return "R32G32B32_FLOAT";
        case DXGI_FORMAT_R32G32B32_UINT: return "R32G32B32_UINT";
        case DXGI_FORMAT_R32G32B32_SINT: return "R32G32B32_SINT";
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return "R16G16B16A16_TYPELESS";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R16G16B16A16_UNORM: return "R16G16B16A16_UNORM";
        case DXGI_FORMAT_R16G16B16A16_UINT: return "R16G16B16A16_UINT";
        case DXGI_FORMAT_R16G16B16A16_SNORM: return "R16G16B16A16_SNORM";
        case DXGI_FORMAT_R16G16B16A16_SINT: return "R16G16B16A16_SINT";
        case DXGI_FORMAT_R32G32_TYPELESS: return "R32G32_TYPELESS";
        case DXGI_FORMAT_R32G32_FLOAT: return "R32G32_FLOAT";
        case DXGI_FORMAT_R32G32_UINT: return "R32G32_UINT";
        case DXGI_FORMAT_R32G32_SINT: return "R32G32_SINT";
        case DXGI_FORMAT_R32G8X24_TYPELESS: return "R32G8X24_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return "D32_FLOAT_S8X24_UINT";
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return "R32_FLOAT_X8X24_TYPELESS";
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: return "X32_TYPELESS_G8X24_UINT";
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return "R10G10B10A2_TYPELESS";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_R10G10B10A2_UINT: return "R10G10B10A2_UINT";
        case DXGI_FORMAT_R11G11B10_FLOAT: return "R11G11B10_FLOAT";
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return "R8G8B8A8_TYPELESS";
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_UINT: return "R8G8B8A8_UINT";
        case DXGI_FORMAT_R8G8B8A8_SNORM: return "R8G8B8A8_SNORM";
        case DXGI_FORMAT_R8G8B8A8_SINT: return "R8G8B8A8_SINT";
        case DXGI_FORMAT_R16G16_TYPELESS: return "R16G16_TYPELESS";
        case DXGI_FORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
        case DXGI_FORMAT_R16G16_UNORM: return "R16G16_UNORM";
        case DXGI_FORMAT_R16G16_UINT: return "R16G16_UINT";
        case DXGI_FORMAT_R16G16_SNORM: return "R16G16_SNORM";
        case DXGI_FORMAT_R16G16_SINT: return "R16G16_SINT";
        case DXGI_FORMAT_R32_TYPELESS: return "R32_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT: return "D32_FLOAT";
        case DXGI_FORMAT_R32_FLOAT: return "R32_FLOAT";
        case DXGI_FORMAT_R32_UINT: return "R32_UINT";
        case DXGI_FORMAT_R32_SINT: return "R32_SINT";
        case DXGI_FORMAT_R24G8_TYPELESS: return "R24G8_TYPELESS";
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return "D24_UNORM_S8_UINT";
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return "R24_UNORM_X8_TYPELESS";
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT: return "X24_TYPELESS_G8_UINT";
        case DXGI_FORMAT_R8G8_TYPELESS: return "R8G8_TYPELESS";
        case DXGI_FORMAT_R8G8_UNORM: return "R8G8_UNORM";
        case DXGI_FORMAT_R8G8_UINT: return "R8G8_UINT";
        case DXGI_FORMAT_R8G8_SNORM: return "R8G8_SNORM";
        case DXGI_FORMAT_R8G8_SINT: return "R8G8_SINT";
        case DXGI_FORMAT_R16_TYPELESS: return "R16_TYPELESS";
        case DXGI_FORMAT_R16_FLOAT: return "R16_FLOAT";
        case DXGI_FORMAT_D16_UNORM: return "D16_UNORM";
        case DXGI_FORMAT_R16_UNORM: return "R16_UNORM";
        case DXGI_FORMAT_R16_UINT: return "R16_UINT";
        case DXGI_FORMAT_R16_SNORM: return "R16_SNORM";
        case DXGI_FORMAT_R16_SINT: return "R16_SINT";
        case DXGI_FORMAT_R8_TYPELESS: return "R8_TYPELESS";
        case DXGI_FORMAT_R8_UNORM: return "R8_UNORM";
        case DXGI_FORMAT_R8_UINT: return "R8_UINT";
        case DXGI_FORMAT_R8_SNORM: return "R8_SNORM";
        case DXGI_FORMAT_R8_SINT: return "R8_SINT";
        case DXGI_FORMAT_A8_UNORM: return "A8_UNORM";
        case DXGI_FORMAT_R1_UNORM: return "R1_UNORM";
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return "R9G9B9E5_SHAREDEXP";
        case DXGI_FORMAT_R8G8_B8G8_UNORM: return "R8G8_B8G8_UNORM";
        case DXGI_FORMAT_G8R8_G8B8_UNORM: return "G8R8_G8B8_UNORM";
        case DXGI_FORMAT_BC1_TYPELESS: return "BC1_TYPELESS";
        case DXGI_FORMAT_BC1_UNORM: return "BC1_UNORM (DXT1)";
        case DXGI_FORMAT_BC1_UNORM_SRGB: return "BC1_UNORM_SRGB";
        case DXGI_FORMAT_BC2_TYPELESS: return "BC2_TYPELESS";
        case DXGI_FORMAT_BC2_UNORM: return "BC2_UNORM (DXT3)";
        case DXGI_FORMAT_BC2_UNORM_SRGB: return "BC2_UNORM_SRGB";
        case DXGI_FORMAT_BC3_TYPELESS: return "BC3_TYPELESS";
        case DXGI_FORMAT_BC3_UNORM: return "BC3_UNORM (DXT5)";
        case DXGI_FORMAT_BC3_UNORM_SRGB: return "BC3_UNORM_SRGB";
        case DXGI_FORMAT_BC4_TYPELESS: return "BC4_TYPELESS";
        case DXGI_FORMAT_BC4_UNORM: return "BC4_UNORM (ATI1)";
        case DXGI_FORMAT_BC4_SNORM: return "BC4_SNORM";
        case DXGI_FORMAT_BC5_TYPELESS: return "BC5_TYPELESS";
        case DXGI_FORMAT_BC5_UNORM: return "BC5_UNORM (ATI2)";
        case DXGI_FORMAT_BC5_SNORM: return "BC5_SNORM";
        case DXGI_FORMAT_B5G6R5_UNORM: return "B5G6R5_UNORM";
        case DXGI_FORMAT_B5G5R5A1_UNORM: return "B5G5R5A1_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8X8_UNORM: return "B8G8R8X8_UNORM";
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: return "R10G10B10_XR_BIAS_A2_UNORM";
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return "B8G8R8A8_TYPELESS";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return "B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_B8G8R8X8_TYPELESS: return "B8G8R8X8_TYPELESS";
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return "B8G8R8X8_UNORM_SRGB";
        case DXGI_FORMAT_BC6H_TYPELESS: return "BC6H_TYPELESS";
        case DXGI_FORMAT_BC6H_UF16: return "BC6H_UF16";
        case DXGI_FORMAT_BC6H_SF16: return "BC6H_SF16";
        case DXGI_FORMAT_BC7_TYPELESS: return "BC7_TYPELESS";
        case DXGI_FORMAT_BC7_UNORM: return "BC7_UNORM";
        case DXGI_FORMAT_BC7_UNORM_SRGB: return "BC7_UNORM_SRGB";
        case DXGI_FORMAT_B4G4R4A4_UNORM: return "B4G4R4A4_UNORM";
        default: {
            std::ostringstream oss;
            oss << "FORMAT_CODE_0x" << std::hex << std::uppercase << static_cast<int>(format);
            return oss.str();
        }
    }
}

// =======================================================================================
// Python Module Definition
// =======================================================================================

PYBIND11_MODULE(directxtex_decoder, m) {
    m.doc() = "High-performance DDS decoder using modern DirectXTex library. Supports legacy, console, and HDR formats.";

    m.def("decode_dds", &decode_dds,
          "Decodes a DDS file from bytes.",
          py::arg("dds_bytes"),
          py::arg("mip_level") = 0,
          py::arg("array_index") = -1,
          py::arg("force_rgba8") = false);

    m.def("get_dds_metadata", &get_dds_metadata,
          "Extracts DDS metadata without decoding pixel data.",
          py::arg("dds_bytes"));

    m.attr("__version__") = "17.0.0";
}
