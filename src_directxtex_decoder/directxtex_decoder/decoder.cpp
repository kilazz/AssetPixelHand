// =======================================================================================
//  decoder.cpp - High-Performance C++ Python Module for DDS File Decoding
// =======================================================================================
// PRODUCTION VERSION 15.1: COMPILER COMPATIBILITY FIX
//
// This version resolves compilation errors by replacing ambiguous initializer-list
// constructors for py::array_t with the explicit std::vector<py::ssize_t> form,
// ensuring maximum compiler compatibility. The core "intelligent channel mapping"
// logic has been significantly expanded to support more DXGI formats.
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
#include <limits> // Required for std::numeric_limits

#define NOMINMAX
#include <Windows.h>
#include <objbase.h>
#include "DirectXTex.h"

namespace py = pybind11;
using namespace DirectX;

// =======================================================================================
// Constants and Macros
// =======================================================================================

// Guard against redefinition from Windows headers
#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3) \
    ((uint32_t)(uint8_t)(ch0) | ((uint32_t)(uint8_t)(ch1) << 8) | \
    ((uint32_t)(uint8_t)(ch2) << 16) | ((uint32_t)(uint8_t)(ch3) << 24))
#endif

// DDS pixel format flags
constexpr uint32_t DDPF_ALPHAPIXELS = 0x00000001;
constexpr uint32_t DDPF_ALPHA = 0x00000002;
constexpr uint32_t DDPF_FOURCC = 0x00000004;
constexpr uint32_t DDPF_RGB = 0x00000040;
constexpr uint32_t DDPF_LUMINANCE = 0x00020000;

// DDS header flags
constexpr uint32_t DDSD_CAPS = 0x00000001;
constexpr uint32_t DDSD_HEIGHT = 0x00000002;
constexpr uint32_t DDSD_WIDTH = 0x00000004;
constexpr uint32_t DDSD_PITCH = 0x00000008;
constexpr uint32_t DDSD_PIXELFORMAT = 0x00001000;
constexpr uint32_t DDSD_MIPMAPCOUNT = 0x00020000;
constexpr uint32_t DDSD_LINEARSIZE = 0x00080000;
constexpr uint32_t DDSD_DEPTH = 0x00800000;

// DDS caps2 flags
constexpr uint32_t DDSCAPS2_CUBEMAP = 0x00000200;
constexpr uint32_t DDSCAPS2_VOLUME = 0x00200000;

// Standard FourCC codes
constexpr uint32_t FOURCC_DXT1 = MAKEFOURCC('D', 'X', 'T', '1');
constexpr uint32_t FOURCC_DXT2 = MAKEFOURCC('D', 'X', 'T', '2');
constexpr uint32_t FOURCC_DXT3 = MAKEFOURCC('D', 'X', 'T', '3');
constexpr uint32_t FOURCC_DXT4 = MAKEFOURCC('D', 'X', 'T', '4');
constexpr uint32_t FOURCC_DXT5 = MAKEFOURCC('D', 'X', 'T', '5');
constexpr uint32_t FOURCC_DX10 = MAKEFOURCC('D', 'X', '1', '0');
constexpr uint32_t FOURCC_ATI1 = MAKEFOURCC('A', 'T', 'I', '1');
constexpr uint32_t FOURCC_BC4U = MAKEFOURCC('B', 'C', '4', 'U');
constexpr uint32_t FOURCC_BC4S = MAKEFOURCC('B', 'C', '4', 'S');
constexpr uint32_t FOURCC_ATI2 = MAKEFOURCC('A', 'T', 'I', '2');
constexpr uint32_t FOURCC_BC5U = MAKEFOURCC('B', 'C', '5', 'U');
constexpr uint32_t FOURCC_BC5S = MAKEFOURCC('B', 'C', '5', 'S');

// Xbox360 FourCC codes
constexpr uint32_t FOURCC_DXT1_XBOX = MAKEFOURCC('1', 'T', 'X', 'D');
constexpr uint32_t FOURCC_DXT3_XBOX = MAKEFOURCC('3', 'T', 'X', 'D');
constexpr uint32_t FOURCC_DXT5_XBOX = MAKEFOURCC('5', 'T', 'X', 'D');
constexpr uint32_t FOURCC_DX10_XBOX = MAKEFOURCC('0', '1', 'X', 'D');

// CryEngine marker
constexpr uint32_t FOURCC_CRYF = MAKEFOURCC('C', 'R', 'Y', 'F');

// HRESULT codes (renamed to avoid conflicts with Windows SDK)
constexpr HRESULT CUSTOM_ERROR_NOT_SUPPORTED = static_cast<HRESULT>(0x80070032);
constexpr HRESULT CUSTOM_ERROR_INVALID_DATA = static_cast<HRESULT>(0x80070026);
constexpr HRESULT CUSTOM_RPC_E_CHANGED_MODE = static_cast<HRESULT>(0x80010106);

// =======================================================================================
// DDS Structures
// =======================================================================================

#pragma pack(push, 1)
struct DDS_PIXELFORMAT {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
};

struct DDS_HEADER {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth;
    uint32_t dwMipMapCount;
    uint32_t dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t dwCaps;
    uint32_t dwCaps2;
    uint32_t dwCaps3;
    uint32_t dwCaps4;
    uint32_t dwReserved2;
};

struct DDS_HEADER_DXT10 {
    DXGI_FORMAT dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlags2;
};
#pragma pack(pop)

// Forward declaration
std::string DXGIFormatToString(DXGI_FORMAT format);

// =======================================================================================
// Utility Functions
// =======================================================================================

template<typename T>
T SwapBytes(T value) {
    static_assert(std::is_integral<T>::value, "SwapBytes can only be used with integral types");
    T result{};
    uint8_t* src = reinterpret_cast<uint8_t*>(&value);
    uint8_t* dst = reinterpret_cast<uint8_t*>(&result);
    for (size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = src[sizeof(T) - 1 - i];
    }
    return result;
}

void EndianSwapBuffer(uint8_t* data, size_t dataSize, size_t elementSize) {
    if (elementSize <= 1) return;
    for (size_t i = 0; i + elementSize <= dataSize; i += elementSize) {
        for (size_t j = 0; j < elementSize / 2; ++j) {
            std::swap(data[i + j], data[i + elementSize - 1 - j]);
        }
    }
}

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
    DWORD result = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&msg_buf), 0, nullptr);
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
        } else { // Subnormal numbers
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x0400;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) { // Infinity
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else { // NaN
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;

    uint32_t result_bits = (sign << 31) | (exponent << 23) | mantissa;
    float result;
    memcpy(&result, &result_bits, sizeof(result));
    return result;
}

// =======================================================================================
// Xbox 360 & CryEngine Specific Logic
// =======================================================================================

bool IsXbox360Format(uint32_t fourCC) {
    switch (fourCC) {
        case FOURCC_DXT1_XBOX:
        case FOURCC_DXT3_XBOX:
        case FOURCC_DXT5_XBOX:
        case FOURCC_DX10_XBOX:
        case MAKEFOURCC('1', 'I', 'T', 'A'):
        case MAKEFOURCC('2', 'I', 'T', 'A'):
        case MAKEFOURCC('U', '4', 'C', 'B'):
        case MAKEFOURCC('U', '5', 'C', 'B'):
            return true;
        default:
            return false;
    }
}

uint32_t Xbox360ToStandardFourCC(uint32_t fourCC) {
    switch (fourCC) {
        case FOURCC_DXT1_XBOX: return FOURCC_DXT1;
        case FOURCC_DXT3_XBOX: return FOURCC_DXT3;
        case FOURCC_DXT5_XBOX: return FOURCC_DXT5;
        case FOURCC_DX10_XBOX: return FOURCC_DX10;
        case MAKEFOURCC('1', 'I', 'T', 'A'): return FOURCC_ATI1;
        case MAKEFOURCC('2', 'I', 'T', 'A'): return FOURCC_ATI2;
        case MAKEFOURCC('U', '4', 'C', 'B'): return FOURCC_BC4U;
        case MAKEFOURCC('U', '5', 'C', 'B'): return FOURCC_BC5U;
        default: return fourCC;
    }
}

void PerformXboxEndianSwap(uint8_t* data, size_t dataSize, DXGI_FORMAT format) {
    if (!data || dataSize == 0) return;
    if (IsCompressed(format)) {
        size_t blockSize = (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC4_UNORM) ? 8 : 16;
        for (size_t i = 0; i + blockSize <= dataSize; i += blockSize) {
            uint16_t* block16 = reinterpret_cast<uint16_t*>(data + i);
            switch (format) {
                case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC4_UNORM:
                    block16[0] = SwapBytes(block16[0]); block16[1] = SwapBytes(block16[1]); break;
                case DXGI_FORMAT_BC2_UNORM:
                    block16[4] = SwapBytes(block16[4]); block16[5] = SwapBytes(block16[5]); break;
                case DXGI_FORMAT_BC3_UNORM:
                    block16[4] = SwapBytes(block16[4]); block16[5] = SwapBytes(block16[5]);
                    block16[1] = SwapBytes(block16[1]); block16[2] = SwapBytes(block16[2]); block16[3] = SwapBytes(block16[3]); break;
                case DXGI_FORMAT_BC5_UNORM:
                    block16[1] = SwapBytes(block16[1]); block16[2] = SwapBytes(block16[2]); block16[3] = SwapBytes(block16[3]);
                    block16[5] = SwapBytes(block16[5]); block16[6] = SwapBytes(block16[6]); block16[7] = SwapBytes(block16[7]); break;
                default: break;
            }
        }
    } else {
        size_t bpp = BitsPerPixel(format);
        if (bpp >= 16) {
            EndianSwapBuffer(data, dataSize, bpp / 8);
        }
    }
}

void UnswizzleBlockLinear(const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height, uint32_t blockBytes, size_t srcSize, size_t dstSize) {
    uint32_t blockWidth = (width + 3) / 4;
    uint32_t blockHeight = (height + 3) / 4;
    size_t requiredSize = static_cast<size_t>(blockWidth) * blockHeight * blockBytes;
    if (srcSize < requiredSize || dstSize < requiredSize) {
        throw std::runtime_error("Buffer too small for unswizzle operation.");
    }

    uint32_t logW = (blockWidth > 1) ? static_cast<uint32_t>(floor(log2(blockWidth - 1))) + 1 : 0;
    uint32_t logH = (blockHeight > 1) ? static_cast<uint32_t>(floor(log2(blockHeight - 1))) + 1 : 0;

    for (uint32_t y = 0; y < blockHeight; ++y) {
        for (uint32_t x = 0; x < blockWidth; ++x) {
            uint32_t swizzledIndex = 0;
            uint32_t min_log = (logW < logH) ? logW : logH;
            for (uint32_t i = 0; i < min_log; ++i) {
                swizzledIndex |= ((x >> i) & 1) << (2 * i);
                swizzledIndex |= ((y >> i) & 1) << (2 * i + 1);
            }
            if (logW > logH) {
                for (uint32_t i = min_log; i < logW; ++i) { swizzledIndex |= ((x >> i) & 1) << (i + min_log); }
            } else {
                for (uint32_t i = min_log; i < logH; ++i) { swizzledIndex |= ((y >> i) & 1) << (i + min_log); }
            }
            size_t srcOffset = static_cast<size_t>(swizzledIndex) * blockBytes;
            size_t dstOffset = (static_cast<size_t>(y) * blockWidth + x) * blockBytes;
            if (srcOffset + blockBytes <= srcSize && dstOffset + blockBytes <= dstSize) {
                memcpy(dst + dstOffset, src + srcOffset, blockBytes);
            }
        }
    }
}

// =======================================================================================
// DDS Header Parsing
// =======================================================================================

DXGI_FORMAT GetDXGIFormatFromHeader(const DDS_HEADER& header, const uint8_t* file_end, bool isXbox360) {
    if (header.ddspf.dwSize != sizeof(DDS_PIXELFORMAT)) return DXGI_FORMAT_UNKNOWN;
    const DDS_PIXELFORMAT& pf = header.ddspf;
    if (pf.dwFlags & DDPF_FOURCC) {
        uint32_t fourCC = isXbox360 ? Xbox360ToStandardFourCC(pf.dwFourCC) : pf.dwFourCC;
        switch (fourCC) {
            case FOURCC_DXT1: return DXGI_FORMAT_BC1_UNORM;
            case FOURCC_DXT2: case FOURCC_DXT3: return DXGI_FORMAT_BC2_UNORM;
            case FOURCC_DXT4: case FOURCC_DXT5: return DXGI_FORMAT_BC3_UNORM;
            case FOURCC_ATI1: case FOURCC_BC4U: return DXGI_FORMAT_BC4_UNORM;
            case FOURCC_BC4S: return DXGI_FORMAT_BC4_SNORM;
            case FOURCC_ATI2: case FOURCC_BC5U: return DXGI_FORMAT_BC5_UNORM;
            case FOURCC_BC5S: return DXGI_FORMAT_BC5_SNORM;
            case FOURCC_DX10: {
                const auto* ext = reinterpret_cast<const DDS_HEADER_DXT10*>(reinterpret_cast<const uint8_t*>(&header) + sizeof(DDS_HEADER));
                if (reinterpret_cast<const uint8_t*>(ext) + sizeof(DDS_HEADER_DXT10) > file_end) return DXGI_FORMAT_UNKNOWN;
                return ext->dxgiFormat;
            }
            default: return static_cast<DXGI_FORMAT>(pf.dwFourCC);
        }
    }
    if (pf.dwFlags & DDPF_RGB) {
        switch (pf.dwRGBBitCount) {
            case 32:
                if (pf.dwRBitMask == 0x00ff0000 && pf.dwGBitMask == 0x0000ff00 && pf.dwBBitMask == 0x000000ff) return DXGI_FORMAT_R8G8B8A8_UNORM;
                if (pf.dwRBitMask == 0x000000ff && pf.dwGBitMask == 0x0000ff00 && pf.dwBBitMask == 0x00ff0000) return DXGI_FORMAT_B8G8R8A8_UNORM;
                break;
            case 24:
                if (pf.dwRBitMask == 0x00ff0000 && pf.dwGBitMask == 0x0000ff00 && pf.dwBBitMask == 0x000000ff) return DXGI_FORMAT_R8G8B8A8_UNORM;
                break;
            case 16:
                if (pf.dwRBitMask == 0xf800 && pf.dwGBitMask == 0x07e0 && pf.dwBBitMask == 0x001f) return DXGI_FORMAT_B5G6R5_UNORM;
                if (pf.dwRBitMask == 0x7c00 && pf.dwGBitMask == 0x03e0 && pf.dwBBitMask == 0x001f) return DXGI_FORMAT_B5G5R5A1_UNORM;
                if (pf.dwRBitMask == 0x0f00 && pf.dwGBitMask == 0x00f0 && pf.dwBBitMask == 0x000f) return DXGI_FORMAT_B4G4R4A4_UNORM;
                break;
        }
    }

    if ((pf.dwFlags & DDPF_ALPHAPIXELS) && pf.dwRGBBitCount == 8) {
        if (pf.dwRBitMask == 0 && pf.dwGBitMask == 0 && pf.dwBBitMask == 0 && pf.dwABitMask == 0xff) {
            return DXGI_FORMAT_A8_UNORM;
        }
    }

    if ((pf.dwFlags & DDPF_LUMINANCE) && pf.dwRGBBitCount == 8) return DXGI_FORMAT_R8_UNORM;
    return DXGI_FORMAT_UNKNOWN;
}

// =======================================================================================
// NumPy Array Creation Helper (The "Smart" Converter) - V2 (Robust Version)
// =======================================================================================
py::object CreateNumpyArrayFromImage(const DirectX::Image* image) {
    if (!image || !image->pixels) {
        throw std::runtime_error("Cannot create NumPy array from an invalid image.");
    }

    // --- Helper lambdas to reduce code duplication ---

    // Helper for direct memory copy into a new NumPy array of a specific type
    auto direct_copy = [&](auto dtype_example, int channels) -> py::object {
        using T = decltype(dtype_example);
        std::vector<py::ssize_t> shape = {(py::ssize_t)image->height, (py::ssize_t)image->width};
        if (channels > 0) {
            shape.push_back(channels);
        }
        auto arr = py::array_t<T>(shape);
        py::buffer_info buf = arr.request();
        // Check if row pitches match for a single memcpy, otherwise copy row by row
        size_t numpy_row_pitch = image->width * sizeof(T) * (channels > 0 ? channels : 1);
        if (image->rowPitch == numpy_row_pitch) {
            memcpy(buf.ptr, image->pixels, image->slicePitch);
        } else {
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for(size_t y = 0; y < image->height; ++y) {
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
        float* dst_ptr = static_cast<float*>(buf.ptr);
        const uint8_t* src_ptr_base = image->pixels;

        for (size_t y = 0; y < image->height; ++y) {
            const uint16_t* src_row_ptr = reinterpret_cast<const uint16_t*>(src_ptr_base + y * image->rowPitch);
            float* dst_row_ptr = dst_ptr + y * image->width * num_components;
            for (size_t i = 0; i < image->width * num_components; ++i) {
                dst_row_ptr[i] = half_to_float(src_row_ptr[i]);
            }
        }
        return py::object(numpy_array);
    };

    // The core logic: decide what to return based on the image format.
    switch (image->format) {
        // --- 32-bit per channel FLOATING POINT formats (HDR) -> returns float32 array ---
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return direct_copy(float(), 4);
        case DXGI_FORMAT_R32G32B32_FLOAT:    return direct_copy(float(), 3);
        case DXGI_FORMAT_R32G32_FLOAT:       return direct_copy(float(), 2);
        case DXGI_FORMAT_R32_FLOAT:          return direct_copy(float(), 0);

        // --- 16-bit per channel FLOATING POINT formats (Half/HDR) -> returns float32 array ---
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return convert_half_to_float(4);
        case DXGI_FORMAT_R16G16_FLOAT:       return convert_half_to_float(2);
        case DXGI_FORMAT_R16_FLOAT:          return convert_half_to_float(1);

        // --- 16-bit per channel INTEGER formats -> returns uint16/int16 array ---
        case DXGI_FORMAT_R16G16B16A16_UNORM: return direct_copy(uint16_t(), 4);
        case DXGI_FORMAT_R16G16_UNORM:       return direct_copy(uint16_t(), 2);
        case DXGI_FORMAT_R16_UNORM:          return direct_copy(uint16_t(), 0);
        case DXGI_FORMAT_R16G16B16A16_UINT:  return direct_copy(uint16_t(), 4);
        case DXGI_FORMAT_R16G16_UINT:        return direct_copy(uint16_t(), 2);
        case DXGI_FORMAT_R16_UINT:           return direct_copy(uint16_t(), 0);
        case DXGI_FORMAT_R16G16B16A16_SINT:  return direct_copy(int16_t(), 4);
        case DXGI_FORMAT_R16G16_SINT:        return direct_copy(int16_t(), 2);
        case DXGI_FORMAT_R16_SINT:           return direct_copy(int16_t(), 0);

        // --- 8-bit per channel UNORM formats (SDR) -> returns uint8 array ---
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return direct_copy(uint8_t(), 4);

        // BGRA format -> requires channel swap to RGBA, returns uint8 array
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    dst_row[x*4 + 0] = src_row[x*4 + 2]; // R
                    dst_row[x*4 + 1] = src_row[x*4 + 1]; // G
                    dst_row[x*4 + 2] = src_row[x*4 + 0]; // B
                    dst_row[x*4 + 3] = src_row[x*4 + 3]; // A
                }
            }
            return arr;
        }

        // --- 8-bit per channel INTEGER formats -> returns uint8/int8 array ---
        case DXGI_FORMAT_R8G8B8A8_UINT: return direct_copy(uint8_t(), 4);
        case DXGI_FORMAT_R8G8_UINT:     return direct_copy(uint8_t(), 2);
        case DXGI_FORMAT_R8_UINT:       return direct_copy(uint8_t(), 0);
        case DXGI_FORMAT_R8G8B8A8_SINT: return direct_copy(int8_t(), 4);
        case DXGI_FORMAT_R8G8_SINT:     return direct_copy(int8_t(), 2);
        case DXGI_FORMAT_R8_SINT:       return direct_copy(int8_t(), 0);

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
            float* dst_ptr = static_cast<float*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
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
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint16_t* src_row = reinterpret_cast<const uint16_t*>(src_ptr + y * image->rowPitch);
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint16_t p = src_row[x];
                    dst_row[x*4 + 0] = static_cast<uint8_t>(((p >> 11) & 0x1F) * 255 / 31); // R
                    dst_row[x*4 + 1] = static_cast<uint8_t>(((p >> 5)  & 0x3F) * 255 / 63); // G
                    dst_row[x*4 + 2] = static_cast<uint8_t>(( p        & 0x1F) * 255 / 31); // B
                    dst_row[x*4 + 3] = 255; // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_B5G5R5A1_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint16_t* src_row = reinterpret_cast<const uint16_t*>(src_ptr + y * image->rowPitch);
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint16_t p = src_row[x];
                    dst_row[x*4 + 0] = static_cast<uint8_t>(((p >> 10) & 0x1F) * 255 / 31); // R
                    dst_row[x*4 + 1] = static_cast<uint8_t>(((p >> 5)  & 0x1F) * 255 / 31); // G
                    dst_row[x*4 + 2] = static_cast<uint8_t>(( p        & 0x1F) * 255 / 31); // B
                    dst_row[x*4 + 3] = static_cast<uint8_t>(((p >> 15) & 0x01) * 255);      // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_B4G4R4A4_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint16_t* src_row = reinterpret_cast<const uint16_t*>(src_ptr + y * image->rowPitch);
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint16_t p = src_row[x];
                    uint8_t r = (p >> 8) & 0x0F, g = (p >> 4) & 0x0F, b = p & 0x0F, a = (p >> 12) & 0x0F;
                    dst_row[x*4 + 0] = (r << 4) | r; // R
                    dst_row[x*4 + 1] = (g << 4) | g; // G
                    dst_row[x*4 + 2] = (b << 4) | b; // B
                    dst_row[x*4 + 3] = (a << 4) | a; // A
                }
            }
            return arr;
        }

        // --- Single and Dual channel UNORM formats -> expand to RGBA8 for visualization ---
        case DXGI_FORMAT_R8G8_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    dst_row[x*4 + 0] = src_row[x*2 + 0]; // R
                    dst_row[x*4 + 1] = src_row[x*2 + 1]; // G
                    dst_row[x*4 + 2] = 0;                // B
                    dst_row[x*4 + 3] = 255;              // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_R8_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    uint8_t val = src_row[x];
                    dst_row[x*4 + 0] = val; // R
                    dst_row[x*4 + 1] = val; // G
                    dst_row[x*4 + 2] = val; // B
                    dst_row[x*4 + 3] = 255; // A
                }
            }
            return arr;
        }
        case DXGI_FORMAT_A8_UNORM: {
            auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)image->height, (py::ssize_t)image->width, 4});
            py::buffer_info buf = arr.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf.ptr);
            const uint8_t* src_ptr = image->pixels;
            for (size_t y = 0; y < image->height; ++y) {
                const uint8_t* src_row = src_ptr + y * image->rowPitch;
                uint8_t* dst_row = dst_ptr + y * image->width * 4;
                for (size_t x = 0; x < image->width; ++x) {
                    dst_row[x*4 + 0] = 255;      // R
                    dst_row[x*4 + 1] = 255;      // G
                    dst_row[x*4 + 2] = 255;      // B
                    dst_row[x*4 + 3] = src_row[x]; // A
                }
            }
            return arr;
        }

        // Default fallback for any other unhandled format
        default:
            throw std::runtime_error("Unsupported DXGI_FORMAT for intelligent NumPy conversion: " + DXGIFormatToString(image->format));
    }
}


// =======================================================================================
// Main DDS Decoding Function
// =======================================================================================
py::dict decode_dds(const py::bytes& dds_bytes, size_t mip_level = 0, py::ssize_t array_index = -1, bool force_rgba8 = false) {
    py::buffer_info info(py::buffer(dds_bytes).request());
    if (info.size < sizeof(DDS_HEADER) + 4) {
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

    const uint8_t* dds_data_ptr = static_cast<const uint8_t*>(info.ptr);
    const size_t dds_data_size = static_cast<size_t>(info.size);
    auto image = std::make_unique<ScratchImage>();
    std::vector<uint8_t> persistent_buffer;

    HRESULT hr = LoadFromDDSMemory(dds_data_ptr, dds_data_size, DDS_FLAGS_NONE, nullptr, *image);

    if (FAILED(hr)) {
        if (hr != CUSTOM_ERROR_NOT_SUPPORTED && hr != CUSTOM_ERROR_INVALID_DATA) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("DirectXTex failed to load DDS memory: " + HResultToString(hr));
        }

        const auto* header = reinterpret_cast<const DDS_HEADER*>(dds_data_ptr + 4);
        const uint8_t* file_end = dds_data_ptr + dds_data_size;

        if (header->dwSize != 124) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("Invalid DDS header size. Expected 124, got " + std::to_string(header->dwSize));
        }

        bool isXbox360 = IsXbox360Format(header->ddspf.dwFourCC);
        DXGI_FORMAT manual_format = GetDXGIFormatFromHeader(*header, file_end, isXbox360);

        if (isXbox360 && header->ddspf.dwFourCC == FOURCC_DX10_XBOX) {
            const auto* ext = reinterpret_cast<const DDS_HEADER_DXT10*>(reinterpret_cast<const uint8_t*>(header) + sizeof(DDS_HEADER));
            if (reinterpret_cast<const uint8_t*>(ext) + sizeof(DDS_HEADER_DXT10) <= file_end) {
                manual_format = static_cast<DXGI_FORMAT>(SwapBytes(static_cast<uint32_t>(ext->dxgiFormat)));
            }
        }

        if (manual_format == DXGI_FORMAT_UNKNOWN) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("Could not determine DDS format from header in fallback mode");
        }

        const uint8_t* image_data_start = dds_data_ptr + 4 + sizeof(DDS_HEADER);
        if (header->ddspf.dwFourCC == FOURCC_DX10 || header->ddspf.dwFourCC == FOURCC_DX10_XBOX) {
            image_data_start += sizeof(DDS_HEADER_DXT10);
        }
        bool isCryEngine = (image_data_start + 8 <= file_end) && (*reinterpret_cast<const uint32_t*>(image_data_start) == FOURCC_CRYF);
        if (isCryEngine) {
            image_data_start += 8;
        }
        if (image_data_start >= file_end) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("Invalid DDS file structure: image data offset is out of bounds");
        }

        size_t row_pitch, slice_pitch;
        hr = ComputePitch(manual_format, header->dwWidth, header->dwHeight, row_pitch, slice_pitch);
        if (FAILED(hr)) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("ComputePitch failed: " + HResultToString(hr));
        }
        if (static_cast<size_t>(file_end - image_data_start) < slice_pitch) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("Insufficient pixel data in file for specified dimensions");
        }

        persistent_buffer.resize(slice_pitch);
        uint8_t* mutable_pixel_data = persistent_buffer.data();
        if (isCryEngine && IsCompressed(manual_format)) {
             try {
                UnswizzleBlockLinear(image_data_start, mutable_pixel_data, header->dwWidth, header->dwHeight, (manual_format == DXGI_FORMAT_BC1_UNORM || manual_format == DXGI_FORMAT_BC4_UNORM) ? 8 : 16, file_end - image_data_start, slice_pitch);
            } catch (const std::exception& e) {
                py::gil_scoped_acquire gil_acquire;
                throw std::runtime_error(std::string("Unswizzle failed: ") + e.what());
            }
        } else {
            memcpy(mutable_pixel_data, image_data_start, slice_pitch);
        }

        if (isXbox360) {
            PerformXboxEndianSwap(mutable_pixel_data, slice_pitch, manual_format);
        }

        Image manual_image{ header->dwWidth, header->dwHeight, manual_format, row_pitch, slice_pitch, mutable_pixel_data };
        hr = image->InitializeFromImage(manual_image);
        if (FAILED(hr)) {
            py::gil_scoped_acquire gil_acquire;
            throw std::runtime_error("Manual processing/initialization failed: " + HResultToString(hr));
        }
    }

    const auto& metadata = image->GetMetadata();
    if (mip_level >= metadata.mipLevels) {
        py::gil_scoped_acquire gil_acquire;
        throw py::value_error("Mipmap level is out of bounds");
    }

    auto process_single_image = [&](const DirectX::Image* input_image) -> std::unique_ptr<ScratchImage> {
        if (!input_image) throw std::runtime_error("Input image for processing is null");
        auto stage1 = std::make_unique<ScratchImage>();
        HRESULT hr_process = stage1->InitializeFromImage(*input_image);
        if (FAILED(hr_process)) throw std::runtime_error("Failed to initialize from selected image: " + HResultToString(hr_process));

        auto current_image = std::move(stage1);
        if (IsCompressed(current_image->GetMetadata().format)) {
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
        if (!selected_image) throw std::runtime_error("Failed to get image slice for index " + std::to_string(current_index));
        processed_images.push_back(std::move(process_single_image(selected_image)));
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
        uint8_t* final_ptr = static_cast<uint8_t*>(final_buf.ptr);

        size_t slice_size_bytes = 1;
        for (size_t dim = 1; dim < final_buf.ndim; ++dim) {
            slice_size_bytes *= final_buf.shape[dim];
        }
        slice_size_bytes *= final_buf.itemsize;

        for (size_t i = 0; i < num_images_to_process; ++i) {
            const DirectX::Image* current_slice = processed_images[i]->GetImage(0, 0, 0);
            uint8_t* dst_slice_ptr = final_ptr + i * slice_size_bytes;
            const uint8_t* src_slice_ptr = current_slice->pixels;
            size_t row_size_bytes = current_slice->width * BitsPerPixel(current_slice->format) / 8;
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
    if (info.size < sizeof(DDS_HEADER) + 4) {
        throw py::value_error("Input data too small for DDS header.");
    }
    if (memcmp(info.ptr, "DDS ", 4) != 0) {
        throw py::value_error("Input is not a valid DDS file.");
    }

    const uint8_t* dds_data_ptr = static_cast<const uint8_t*>(info.ptr);
    const size_t dds_data_size = static_cast<size_t>(info.size);

    TexMetadata metadata;
    {
        py::gil_scoped_release release;
        HRESULT hr = GetMetadataFromDDSMemory(dds_data_ptr, dds_data_size, DDS_FLAGS_NONE, metadata);

        if (FAILED(hr) && (hr == CUSTOM_ERROR_NOT_SUPPORTED || hr == CUSTOM_ERROR_INVALID_DATA)) {
            const DDS_HEADER* header = reinterpret_cast<const DDS_HEADER*>(dds_data_ptr + 4);
            const uint8_t* file_end = dds_data_ptr + dds_data_size;
            bool isXbox360 = IsXbox360Format(header->ddspf.dwFourCC);

            metadata.width = header->dwWidth;
            metadata.height = header->dwHeight;
            metadata.depth = (header->dwFlags & DDSD_DEPTH) ? header->dwDepth : 1;
            metadata.mipLevels = (header->dwFlags & DDSD_MIPMAPCOUNT) ? header->dwMipMapCount : 1;
            metadata.arraySize = (header->dwCaps2 & DDSCAPS2_CUBEMAP) ? 6 : 1;
            metadata.format = GetDXGIFormatFromHeader(*header, file_end, isXbox360);

            if (isXbox360 && header->ddspf.dwFourCC == FOURCC_DX10_XBOX) {
                const auto* ext = reinterpret_cast<const DDS_HEADER_DXT10*>(reinterpret_cast<const uint8_t*>(header) + sizeof(DDS_HEADER));
                if (reinterpret_cast<const uint8_t*>(ext) + sizeof(DDS_HEADER_DXT10) <= file_end) {
                    metadata.format = static_cast<DXGI_FORMAT>(SwapBytes(static_cast<uint32_t>(ext->dxgiFormat)));
                }
            }

            metadata.miscFlags = (header->dwCaps2 & DDSCAPS2_CUBEMAP) ? TEX_MISC_TEXTURECUBE : 0;
            metadata.miscFlags2 = 0;
            if (header->dwCaps2 & DDSCAPS2_VOLUME) {
                metadata.dimension = TEX_DIMENSION_TEXTURE3D;
            } else {
                 metadata.dimension = (header->dwHeight > 0 && header->dwWidth > 0) ? TEX_DIMENSION_TEXTURE2D : TEX_DIMENSION_TEXTURE1D;
            }
        } else if (FAILED(hr)) {
            py::gil_scoped_acquire acquire;
            throw std::runtime_error("Failed to get metadata: " + HResultToString(hr));
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
// Python Module Definition
// =======================================================================================
PYBIND11_MODULE(directxtex_decoder, m) {
    m.doc() = "High-performance DDS decoder with support for CryEngine, Xbox360, HDR, and full structures (cubemaps, arrays).";

    m.def("decode_dds", &decode_dds,
          "Decodes a DDS file. By default (array_index=-1), decodes all images in a structure (e.g., all 6 cubemap faces).",
          py::arg("dds_bytes"),
          py::arg("mip_level") = 0,
          py::arg("array_index") = -1,
          py::arg("force_rgba8") = false);

    m.def("get_dds_metadata", &get_dds_metadata,
          "Extracts DDS metadata without decoding pixel data.",
          py::arg("dds_bytes"));

    m.attr("__version__") = "15.2.0"; // Version updated to reflect significant change
    m.attr("__author__") = "DirectXTex Wrapper";
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
