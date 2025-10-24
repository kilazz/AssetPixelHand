// =======================================================================================
//  decoder.cpp - A High-Performance C++ Python Module for DDS File Decoding
// =======================================================================================
// FINAL PRODUCTION-READY VERSION 8.0: DEFINITIVE FIX
// - Corrects GIL management in get_dds_metadata to prevent race conditions.
// - Corrects NumPy array creation to copy data, preventing memory errors.
// - Retains all CryEngine-specific logic (un-swizzling, uncompressed format handling).
// This is the definitive and stable version.
// =======================================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // REQUIRED for NumPy integration
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <Windows.h>
#include <objbase.h>
#include "DirectXTex.h"

// Structs for manual header parsing, must match DDS spec
#pragma pack(push,1)
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

struct DDS_HEADER_DXT10 {
    DXGI_FORMAT dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlags2;
};

struct DDS_HEADER {
    uint32_t dwSize, dwFlags, dwHeight, dwWidth, dwPitchOrLinearSize, dwDepth, dwMipMapCount;
    uint32_t dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t dwCaps, dwCaps2, dwCaps3, dwCaps4, dwReserved2;
};
#pragma pack(pop)

namespace py = pybind11;

std::string DXGIFormatToString(DXGI_FORMAT format); // Forward declaration

// Un-swizzling function for CryEngine compressed textures (ported from CryConvertXDDS)
void UnswizzleBlockLinear(const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height, uint32_t blockBytes) {
    uint32_t blockWidth = (width + 3) / 4;
    uint32_t blockHeight = (height + 3) / 4;
    for (uint32_t y = 0; y < blockHeight; ++y) {
        for (uint32_t x = 0; x < blockWidth; ++x) {
            uint32_t sx = 0, sy = 0, tx = x, ty = y, bit = 0;
            while ((1u << bit) < blockWidth || (1u << bit) < blockHeight) {
                if ((1u << bit) < blockWidth) { sx |= (tx & 1) << bit; tx >>= 1; }
                if ((1u << bit) < blockHeight) { sy |= (ty & 1) << bit; ty >>= 1; }
                bit++;
            }
            sx |= tx << bit; sy |= ty << bit;
            uint32_t srcOffset = (sy * blockWidth + sx) * blockBytes;
            uint32_t dstOffset = (y * blockWidth + x) * blockBytes;
            if (srcOffset < blockWidth * blockHeight * blockBytes && dstOffset < blockWidth * blockHeight * blockBytes) {
                memcpy(dst + dstOffset, src + srcOffset, blockBytes);
            }
        }
    }
}

// Universal header parser for standard and CryEngine uncompressed formats
DXGI_FORMAT GetDXGIFormatFromHeader(const DDS_HEADER& header, const uint8_t* file_end) {
    const DDS_PIXELFORMAT& pf = header.ddspf;
    if (pf.dwFlags & 0x4) { // DDPF_FOURCC
        switch (pf.dwFourCC) {
            case '1TXD': return DXGI_FORMAT_BC1_UNORM;
            case '3TXD': return DXGI_FORMAT_BC2_UNORM;
            case '5TXD': return DXGI_FORMAT_BC3_UNORM;
            case '1ITA': case 'U4CB': return DXGI_FORMAT_BC4_UNORM;
            case '2ITA': case 'U5CB': return DXGI_FORMAT_BC5_UNORM;
            case '01XD': {
                const auto* ext = reinterpret_cast<const DDS_HEADER_DXT10*>(reinterpret_cast<const uint8_t*>(&header) + sizeof(DDS_HEADER));
                if (reinterpret_cast<const uint8_t*>(ext) + sizeof(DDS_HEADER_DXT10) > file_end) return DXGI_FORMAT_UNKNOWN;
                return ext->dxgiFormat;
            }
            default: return static_cast<DXGI_FORMAT>(pf.dwFourCC);
        }
    }
    if (pf.dwFlags & 0x40) { // DDPF_RGB
        if (pf.dwRGBBitCount == 32) {
            if (pf.dwRBitMask == 0x00FF0000 && pf.dwGBitMask == 0x0000FF00 && pf.dwBBitMask == 0x000000FF)
                if (pf.dwABitMask == 0xFF000000 || pf.dwABitMask == 0) return DXGI_FORMAT_R8G8B8A8_UNORM;
            if (pf.dwRBitMask == 0x000000FF && pf.dwGBitMask == 0x0000FF00 && pf.dwBBitMask == 0x00FF0000)
                if (pf.dwABitMask == 0xFF000000 || pf.dwABitMask == 0) return DXGI_FORMAT_B8G8R8A8_UNORM;
        }
    }
    if ((pf.dwFlags & 0x1) && pf.dwRGBBitCount == 8) return DXGI_FORMAT_A8_UNORM;
    return DXGI_FORMAT_UNKNOWN;
}

struct CoInitializer { HRESULT hr; CoInitializer() : hr(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)) {} ~CoInitializer() { if (SUCCEEDED(hr)) CoUninitialize(); } };
std::string HResultToString(HRESULT hr) {
    char* msg_buf = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&msg_buf, 0, NULL);
    std::string msg = msg_buf ? msg_buf : "Unknown HRESULT error";
    if(msg_buf) LocalFree(msg_buf);
    return msg;
}

py::dict decode_dds(const py::bytes& dds_bytes, size_t mip_level = 0, size_t array_index = 0) {
    py::buffer_info info(py::buffer(dds_bytes).request());
    if (info.size < 4 || memcmp(info.ptr, "DDS ", 4) != 0) throw py::value_error("Input is not a valid DDS file.");

    py::gil_scoped_release release;
    CoInitializer co_init;
    if (FAILED(co_init.hr)) { py::gil_scoped_acquire acquire; throw std::runtime_error("COM initialization failed."); }

    const uint8_t* dds_data_ptr = static_cast<const uint8_t*>(info.ptr);
    const size_t dds_data_size = static_cast<size_t>(info.size);
    auto image = std::make_unique<DirectX::ScratchImage>();
    auto temp_image = std::make_unique<DirectX::ScratchImage>();
    HRESULT hr = DirectX::LoadFromDDSMemory(dds_data_ptr, dds_data_size, DirectX::DDS_FLAGS_NONE, nullptr, *image);

    if (FAILED(hr)) {
         if ((hr == 0x80070032 || hr == 0x80070026) && dds_data_size > sizeof(DDS_HEADER) + 4) {
            const auto* header = reinterpret_cast<const DDS_HEADER*>(dds_data_ptr + 4);
            DXGI_FORMAT manual_format = GetDXGIFormatFromHeader(*header, dds_data_ptr + dds_data_size);

            const uint8_t* image_data_start = dds_data_ptr + 4 + sizeof(DDS_HEADER);
            if (header->ddspf.dwFourCC == '01XD') image_data_start += sizeof(DDS_HEADER_DXT10);

            bool isCryEngineFile = (dds_data_size > (image_data_start - dds_data_ptr) + 8 && memcmp(image_data_start, "CRYF", 4) == 0);
            if (isCryEngineFile) image_data_start += 8;

            DirectX::Image manual_image = { header->dwWidth, header->dwHeight, manual_format, 0, 0, const_cast<uint8_t*>(image_data_start) };
            if (manual_image.format == DXGI_FORMAT_UNKNOWN) { py::gil_scoped_acquire acquire; throw std::runtime_error("Manual header parse failed: Could not determine format."); }

            size_t rowPitch, slicePitch;
            DirectX::ComputePitch(manual_image.format, manual_image.width, manual_image.height, rowPitch, slicePitch);
            manual_image.rowPitch = rowPitch; manual_image.slicePitch = slicePitch;

            std::vector<uint8_t> unswizzled_buffer;
            if (DirectX::IsCompressed(manual_image.format) && isCryEngineFile) {
                size_t blockBytes = (manual_format == DXGI_FORMAT_BC1_UNORM || manual_format == DXGI_FORMAT_BC4_UNORM) ? 8 : 16;
                unswizzled_buffer.resize(manual_image.slicePitch);
                UnswizzleBlockLinear(manual_image.pixels, unswizzled_buffer.data(), manual_image.width, manual_image.height, blockBytes);
                manual_image.pixels = unswizzled_buffer.data();
            }

            if (DirectX::IsCompressed(manual_image.format)) hr = DirectX::Decompress(manual_image, DXGI_FORMAT_UNKNOWN, *image);
            else hr = image->InitializeFromImage(manual_image);
            if (FAILED(hr)) { py::gil_scoped_acquire acquire; throw std::runtime_error("Manual fallback processing failed: " + HResultToString(hr)); }
        } else {
            py::gil_scoped_acquire acquire; throw std::runtime_error("Failed to load DDS from memory: " + HResultToString(hr));
        }
    }

    const auto& metadata = image->GetMetadata();
    if (mip_level >= metadata.mipLevels || array_index >= metadata.arraySize) { py::gil_scoped_acquire acquire; throw py::value_error("mip_level or array_index is out of bounds."); }

    const DirectX::Image* selected_image = image->GetImage(mip_level, array_index, 0);
    if (!selected_image) { py::gil_scoped_acquire acquire; throw std::runtime_error("Failed to get image slice."); }

    DirectX::ScratchImage current_slice;
    current_slice.InitializeFromImage(*selected_image);

    DirectX::ScratchImage* pCurrent = &current_slice;
    DirectX::ScratchImage* pTemp = temp_image.get();

    if (DirectX::IsCompressed(pCurrent->GetMetadata().format)) {
        DirectX::Decompress(*pCurrent->GetImage(0, 0, 0), DXGI_FORMAT_UNKNOWN, *pTemp);
        std::swap(pCurrent, pTemp);
    }
    if (pCurrent->GetMetadata().format != DXGI_FORMAT_R8G8B8A8_UNORM && pCurrent->GetMetadata().format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        DirectX::Convert(*pCurrent->GetImage(0, 0, 0), DXGI_FORMAT_R8G8B8A8_UNORM, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, *pTemp);
        std::swap(pCurrent, pTemp);
    }

    const DirectX::Image* final_image = pCurrent->GetImage(0, 0, 0);
    if (!final_image) { py::gil_scoped_acquire acquire; throw std::runtime_error("Failed to get final image data."); }

    py::gil_scoped_acquire acquire;

    std::vector<py::ssize_t> shape = { (py::ssize_t)final_image->height, (py::ssize_t)final_image->width, 4 };
    auto numpy_array = py::array_t<uint8_t>(shape);
    uint8_t* dst_ptr = static_cast<uint8_t*>(numpy_array.mutable_data());
    memcpy(dst_ptr, final_image->pixels, final_image->slicePitch);

    std::string format_name = (final_image->format == DXGI_FORMAT_B8G8R8A8_UNORM) ? "BGRA" : "RGBA";

    return py::dict(
        py::arg("width") = final_image->width,
        py::arg("height") = final_image->height,
        py::arg("data") = numpy_array,
        py::arg("format") = format_name
    );
}

py::dict get_dds_metadata(const py::bytes& dds_bytes) {
    py::buffer_info info(py::buffer(dds_bytes).request());
    if (info.size < 128) { throw py::value_error("Input data too small for DDS header."); }
    if (memcmp(info.ptr, "DDS ", 4) != 0) { throw py::value_error("Input is not a valid DDS file."); }

    DirectX::TexMetadata metadata;
    {
        py::gil_scoped_release release;
        const uint8_t* dds_data_ptr = static_cast<const uint8_t*>(info.ptr);
        const size_t dds_data_size = static_cast<size_t>(info.size);
        const uint8_t* file_end = dds_data_ptr + dds_data_size;
        HRESULT hr = DirectX::GetMetadataFromDDSMemory(dds_data_ptr, dds_data_size, DirectX::DDS_FLAGS_NONE, metadata);

        constexpr HRESULT ERROR_NOT_SUPPORTED_HRESULT = 0x80070032;
        if (FAILED(hr)) {
            if (hr == ERROR_NOT_SUPPORTED_HRESULT && dds_data_size > sizeof(DDS_HEADER) + 4) {
                 const DDS_HEADER* header = reinterpret_cast<const DDS_HEADER*>(dds_data_ptr + 4);
                 metadata.width = header->dwWidth;
                 metadata.height = header->dwHeight;
                 metadata.format = GetDXGIFormatFromHeader(*header, file_end);
                 metadata.mipLevels = header->dwMipMapCount > 0 ? header->dwMipMapCount : 1;
                 metadata.arraySize = (header->dwCaps2 & 0x200) ? header->dwDepth : 1;
                 metadata.depth = (header->dwFlags & 0x800000) ? header->dwDepth : 1;
                 metadata.dimension = (header->dwFlags & 0x800000) ? DirectX::TEX_DIMENSION_TEXTURE3D : ((header->dwCaps2 & 0x200) ? DirectX::TEX_DIMENSION_TEXTURE2D : DirectX::TEX_DIMENSION_TEXTURE2D);
            } else {
                py::gil_scoped_acquire acquire;
                throw std::runtime_error("Failed to get metadata from DDS memory: " + HResultToString(hr));
            }
        }
    }

    // --- FINAL GIL FIX ---
    // Re-acquire the GIL before returning to Python.
    py::gil_scoped_acquire acquire;

    return py::dict(py::arg("width")=metadata.width, py::arg("height")=metadata.height, py::arg("format_str")=DXGIFormatToString(metadata.format), py::arg("mip_levels")=metadata.mipLevels, py::arg("is_cubemap")=metadata.IsCubemap(), py::arg("array_size")=metadata.arraySize, py::arg("is_3d")=(metadata.dimension == DirectX::TEX_DIMENSION_TEXTURE3D));
}

PYBIND11_MODULE(directxtex_decoder, m) {
    m.doc() = "A high-performance C++ module to decode and inspect DDS files using DirectXTex.";
    m.def("decode_dds", &decode_dds, "Decodes a specific MIP/array slice.", py::arg("dds_bytes"), py::arg("mip_level") = 0, py::arg("array_index") = 0);
    m.def("get_dds_metadata", &get_dds_metadata, "Quickly reads DDS header metadata.", py::arg("dds_bytes"));
}

std::string DXGIFormatToString(DXGI_FORMAT format) {
    #pragma region DXGI_FORMAT_ToString_Switch
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
        case DXGI_FORMAT_BC1_UNORM: return "BC1 (DXT1)";
        case DXGI_FORMAT_BC1_UNORM_SRGB: return "BC1_SRGB (DXT1)";
        case DXGI_FORMAT_BC2_TYPELESS: return "BC2_TYPELESS";
        case DXGI_FORMAT_BC2_UNORM: return "BC2 (DXT3)";
        case DXGI_FORMAT_BC2_UNORM_SRGB: return "BC2_SRGB (DXT3)";
        case DXGI_FORMAT_BC3_TYPELESS: return "BC3_TYPELESS";
        case DXGI_FORMAT_BC3_UNORM: return "BC3 (DXT5)";
        case DXGI_FORMAT_BC3_UNORM_SRGB: return "BC3_SRGB (DXT5)";
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
        default:
            std::ostringstream oss;
            oss << "UNKNOWN_FORMAT_CODE_" << format;
            return oss.str();
    }
    #pragma endregion
}