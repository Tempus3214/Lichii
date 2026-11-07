// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/logging.h"
#include "video_core/vulkan_common/vulkan_pipeline_binary_cache.h"

namespace Vulkan::PipelineBinaryCache {
namespace {

// File: magic[8] u32 global_key_size u8 global_key[32], then records:
//   u32 record_magic u32 key_size u8 key[32] u64 payload_size payload
// Payload: u32 count, then per binary: u32 key_size u8 key[32] u64 data_size data
constexpr std::array<char, 8> FILE_MAGIC{'L', 'C', 'H', 'P', 'B', 'I', 'N', '1'};
constexpr u32 RECORD_MAGIC = 0x44524350;
constexpr size_t KEY_BYTES = VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;
constexpr size_t HEADER_BYTES = FILE_MAGIC.size() + 4 + KEY_BYTES;
constexpr size_t RECORD_HEADER_BYTES = 4 + 4 + KEY_BYTES + 8;
constexpr u32 MAX_BINARIES_PER_PIPELINE = 64;

struct Stats {
    std::atomic<u64> loaded{}, compiled{}, saved{}, load_failed{}, skipped{};
    std::atomic<u64> loaded_us{}, compiled_us{};
};

class Store {
public:
    bool Open(VkDevice device_, const vk::DeviceDispatch& dld_, const std::filesystem::path& file) {
        device = device_;
        dld = &dld_;
        path = file;
        VkPipelineBinaryKeyKHR global{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        if (dld->vkGetPipelineKeyKHR(device, nullptr, &global) != VK_SUCCESS ||
            global.keySize == 0 || global.keySize > KEY_BYTES) {
            LOG_WARNING(Render_Vulkan, "Pipeline binaries: the driver gave no global key");
            return false;
        }
        std::array<u8, HEADER_BYTES> header{};
        std::memcpy(header.data(), FILE_MAGIC.data(), FILE_MAGIC.size());
        std::memcpy(header.data() + 8, &global.keySize, 4);
        std::memcpy(header.data() + 12, global.key, global.keySize);

        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        size_t good_end = 0;
        {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            if (in) {
                const auto size = static_cast<size_t>(in.tellg());
                in.seekg(0);
                loaded.resize(size);
                if (!in.read(reinterpret_cast<char*>(loaded.data()), size)) {
                    loaded.clear();
                }
            }
        }
        const bool same_driver = loaded.size() >= HEADER_BYTES &&
                                 std::memcmp(loaded.data(), header.data(), HEADER_BYTES) == 0;
        if (same_driver) {
            size_t p = HEADER_BYTES;
            good_end = p;
            while (p + RECORD_HEADER_BYTES <= loaded.size()) {
                u32 magic, key_size;
                u64 len;
                std::memcpy(&magic, &loaded[p], 4);
                std::memcpy(&key_size, &loaded[p + 4], 4);
                std::memcpy(&len, &loaded[p + 8 + KEY_BYTES], 8);
                if (magic != RECORD_MAGIC || key_size == 0 || key_size > KEY_BYTES ||
                    len > loaded.size() - p - RECORD_HEADER_BYTES) {
                    break;
                }
                index.emplace(std::string(reinterpret_cast<const char*>(&loaded[p + 8]), key_size),
                              Blob{&loaded[p + RECORD_HEADER_BYTES], static_cast<size_t>(len)});
                p += RECORD_HEADER_BYTES + static_cast<size_t>(len);
                good_end = p;
            }
            if (good_end < loaded.size()) {
                // A session that ended mid-write leaves a partial record; drop it.
                std::filesystem::resize_file(path, good_end, ec);
            }
            out.open(path, std::ios::binary | std::ios::app);
        } else {
            // First run, or the driver changed: binaries from another driver can't be used, so the
            // file is replaced rather than kept beside the new one.
            if (!loaded.empty()) {
                LOG_INFO(Render_Vulkan, "Pipeline binaries: driver changed, starting over");
            }
            loaded.clear();
            out.open(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(header.data()), header.size());
            out.flush();
        }
        if (!out) {
            LOG_ERROR(Render_Vulkan, "Pipeline binaries: can't write {}", path.string());
            return false;
        }
        LOG_INFO(Render_Vulkan, "Pipeline binaries: {} saved pipelines ({:.1f} MiB) in {}",
                 index.size(), static_cast<double>(loaded.size()) / (1024.0 * 1024.0),
                 path.string());
        return true;
    }

    void Close() {
        std::unique_lock lock{mutex};
        if (out.is_open()) {
            out.close();
        }
        LOG_INFO(Render_Vulkan,
                 "Pipeline binaries: {} pipelines built from saved binaries (avg {:.2f} ms), {} "
                 "compiled (avg {:.2f} ms), {} saved, {} saved binaries rejected",
                 stats.loaded.load(), Average(stats.loaded_us, stats.loaded),
                 stats.compiled.load(), Average(stats.compiled_us, stats.compiled),
                 stats.saved.load(), stats.load_failed.load());
    }

    template <typename CreateInfo, typename CreateFn>
    bool Create(const CreateInfo& ci, VkPipelineCache cache, VkPipeline* out_pipeline,
                VkResult* result, CreateFn&& create) {
        if (!Eligible(ci)) {
            ++stats.skipped;
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        const auto elapsed_us = [&] {
            return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count());
        };
        const VkPipelineCreateInfoKHR key_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
            .pNext = const_cast<CreateInfo*>(&ci),
        };
        VkPipelineBinaryKeyKHR key{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        if (dld->vkGetPipelineKeyKHR(device, &key_info, &key) != VK_SUCCESS || key.keySize == 0 ||
            key.keySize > KEY_BYTES) {
            ++stats.skipped;
            return false;
        }
        const std::string key_bytes(reinterpret_cast<const char*>(key.key), key.keySize);

        if (const auto blob = Find(key_bytes)) {
            if (CreateFromSaved(*blob, ci, out_pipeline, create)) {
                ++stats.loaded;
                stats.loaded_us += elapsed_us();
                *result = VK_SUCCESS;
                return true;
            }
            if (stats.load_failed++ < 5) {
                LOG_WARNING(Render_Vulkan, "Pipeline binaries: a saved pipeline was rejected, "
                                           "compiling it again");
            }
        }

        // Compile with capture, then keep the driver's binaries.
        CreateInfo capture_ci = ci;
        VkPipelineCreateFlags2CreateInfoKHR flags2{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
            .pNext = ci.pNext,
            .flags = static_cast<VkPipelineCreateFlags2KHR>(ci.flags) |
                     VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR,
        };
        capture_ci.pNext = &flags2;
        VkResult res = create(capture_ci, cache, out_pipeline);
        if (res != VK_SUCCESS) {
            if (res < 0) {
                // Let the caller build it without capture and report its own errors.
                *out_pipeline = VK_NULL_HANDLE;
                return false;
            }
            *result = res;
            return true;
        }
        ++stats.compiled;
        Save(key_bytes, *out_pipeline);
        const VkReleaseCapturedPipelineDataInfoKHR release{
            .sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR,
            .pipeline = *out_pipeline,
        };
        dld->vkReleaseCapturedPipelineDataKHR(device, &release, nullptr);
        stats.compiled_us += elapsed_us();
        *result = VK_SUCCESS;
        return true;
    }

private:
    struct Blob {
        const u8* data;
        size_t size;
    };

    static double Average(const std::atomic<u64>& us, const std::atomic<u64>& count) {
        const u64 n = count.load();
        return n == 0 ? 0.0 : static_cast<double>(us.load()) / 1000.0 / static_cast<double>(n);
    }

    static const VkBaseInStructure* Find(const void* next, VkStructureType type) {
        for (auto* s = static_cast<const VkBaseInStructure*>(next); s; s = s->pNext) {
            if (s->sType == type) {
                return s;
            }
        }
        return nullptr;
    }

    template <typename CreateInfo>
    static bool Eligible(const CreateInfo& ci) {
        // Libraries, pipelines already built from binaries and callers that pass their own
        // VkPipelineCreateFlags2 are left alone.
        return (ci.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) == 0 &&
               !Find(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR) &&
               !Find(ci.pNext, VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT) &&
               !Find(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR) &&
               !Find(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR);
    }

    std::optional<Blob> Find(const std::string& key) {
        std::shared_lock lock{mutex};
        const auto it = index.find(key);
        return it == index.end() ? std::nullopt : std::optional<Blob>{it->second};
    }

    template <typename CreateInfo, typename CreateFn>
    bool CreateFromSaved(const Blob& blob, const CreateInfo& ci, VkPipeline* out_pipeline,
                         CreateFn& create) {
        if (blob.size < 4) {
            return false;
        }
        u32 count;
        std::memcpy(&count, blob.data, 4);
        if (count == 0 || count > MAX_BINARIES_PER_PIPELINE) {
            return false;
        }
        std::vector<VkPipelineBinaryKeyKHR> keys(count);
        std::vector<VkPipelineBinaryDataKHR> datas(count);
        size_t o = 4;
        for (u32 i = 0; i < count; ++i) {
            if (o + 4 + KEY_BYTES + 8 > blob.size) {
                return false;
            }
            keys[i] = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            std::memcpy(&keys[i].keySize, blob.data + o, 4);
            if (keys[i].keySize > KEY_BYTES) {
                return false;
            }
            std::memcpy(keys[i].key, blob.data + o + 4, KEY_BYTES);
            u64 size;
            std::memcpy(&size, blob.data + o + 4 + KEY_BYTES, 8);
            o += 4 + KEY_BYTES + 8;
            if (size > blob.size - o) {
                return false;
            }
            datas[i] = {.dataSize = static_cast<size_t>(size),
                        .pData = const_cast<u8*>(blob.data + o)};
            o += static_cast<size_t>(size);
        }
        const VkPipelineBinaryKeysAndDataKHR keys_and_data{
            .binaryCount = count,
            .pPipelineBinaryKeys = keys.data(),
            .pPipelineBinaryData = datas.data(),
        };
        const VkPipelineBinaryCreateInfoKHR binary_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
            .pKeysAndDataInfo = &keys_and_data,
        };
        std::vector<VkPipelineBinaryKHR> binaries(count, VK_NULL_HANDLE);
        VkPipelineBinaryHandlesInfoKHR handles{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
            .pipelineBinaryCount = count,
            .pPipelineBinaries = binaries.data(),
        };
        VkResult res = dld->vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles);
        if (res == VK_SUCCESS) {
            const VkPipelineBinaryInfoKHR binary_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
                .pNext = ci.pNext,
                .binaryCount = count,
                .pPipelineBinaries = binaries.data(),
            };
            CreateInfo from_binaries = ci;
            from_binaries.pNext = &binary_info;
            // A pipeline built from binaries takes no pipeline cache.
            res = create(from_binaries, VK_NULL_HANDLE, out_pipeline);
        }
        for (const VkPipelineBinaryKHR binary : binaries) {
            if (binary != VK_NULL_HANDLE) {
                dld->vkDestroyPipelineBinaryKHR(device, binary, nullptr);
            }
        }
        if (res != VK_SUCCESS) {
            *out_pipeline = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    void Save(const std::string& key, VkPipeline pipeline) {
        const VkPipelineBinaryCreateInfoKHR binary_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
            .pipeline = pipeline,
        };
        VkPipelineBinaryHandlesInfoKHR handles{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
        };
        if (dld->vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) !=
                VK_SUCCESS ||
            handles.pipelineBinaryCount == 0 ||
            handles.pipelineBinaryCount > MAX_BINARIES_PER_PIPELINE) {
            return;
        }
        std::vector<VkPipelineBinaryKHR> binaries(handles.pipelineBinaryCount, VK_NULL_HANDLE);
        handles.pPipelineBinaries = binaries.data();
        if (dld->vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) !=
            VK_SUCCESS) {
            return;
        }
        std::vector<u8> payload(4);
        const u32 count = static_cast<u32>(binaries.size());
        std::memcpy(payload.data(), &count, 4);
        bool ok = true;
        for (const VkPipelineBinaryKHR binary : binaries) {
            const VkPipelineBinaryDataInfoKHR data_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
                .pipelineBinary = binary,
            };
            VkPipelineBinaryKeyKHR binary_key{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            size_t size = 0;
            if (dld->vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size, nullptr) !=
                    VK_SUCCESS ||
                size == 0) {
                ok = false;
                break;
            }
            const size_t o = payload.size();
            payload.resize(o + 4 + KEY_BYTES + 8 + size);
            std::memcpy(&payload[o], &binary_key.keySize, 4);
            std::memcpy(&payload[o + 4], binary_key.key, KEY_BYTES);
            const u64 size64 = size;
            std::memcpy(&payload[o + 4 + KEY_BYTES], &size64, 8);
            if (dld->vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size,
                                                &payload[o + 4 + KEY_BYTES + 8]) != VK_SUCCESS) {
                ok = false;
                break;
            }
        }
        for (const VkPipelineBinaryKHR binary : binaries) {
            dld->vkDestroyPipelineBinaryKHR(device, binary, nullptr);
        }
        if (!ok) {
            return;
        }
        std::unique_lock lock{mutex};
        if (index.contains(key)) {
            return;
        }
        auto owned = std::make_unique<std::vector<u8>>(std::move(payload));
        if (out.is_open()) {
            const u32 magic = RECORD_MAGIC;
            const u32 key_size = static_cast<u32>(key.size());
            std::array<u8, KEY_BYTES> key_padded{};
            std::memcpy(key_padded.data(), key.data(), key.size());
            const u64 len = owned->size();
            out.write(reinterpret_cast<const char*>(&magic), 4);
            out.write(reinterpret_cast<const char*>(&key_size), 4);
            out.write(reinterpret_cast<const char*>(key_padded.data()), KEY_BYTES);
            out.write(reinterpret_cast<const char*>(&len), 8);
            out.write(reinterpret_cast<const char*>(owned->data()),
                      static_cast<std::streamsize>(owned->size()));
            out.flush();
        }
        index.emplace(key, Blob{owned->data(), owned->size()});
        session_blobs.push_back(std::move(owned));
        ++stats.saved;
    }

    VkDevice device{};
    const vk::DeviceDispatch* dld{};
    std::filesystem::path path;
    std::ofstream out;
    std::vector<u8> loaded;
    std::vector<std::unique_ptr<std::vector<u8>>> session_blobs;
    std::unordered_map<std::string, Blob> index;
    std::shared_mutex mutex;
    Stats stats;
};

std::shared_mutex active_mutex;
std::unique_ptr<Store> active;

} // Anonymous namespace

void Activate(VkDevice device, const vk::DeviceDispatch& dld, const std::filesystem::path& file) {
    if (!dld.vkGetPipelineKeyKHR || !dld.vkCreatePipelineBinariesKHR ||
        !dld.vkDestroyPipelineBinaryKHR || !dld.vkGetPipelineBinaryDataKHR ||
        !dld.vkReleaseCapturedPipelineDataKHR) {
        return;
    }
    auto store = std::make_unique<Store>();
    if (!store->Open(device, dld, file)) {
        return;
    }
    std::unique_lock lock{active_mutex};
    if (active) {
        active->Close();
    }
    active = std::move(store);
}

void Deactivate() {
    std::unique_lock lock{active_mutex};
    if (active) {
        active->Close();
        active.reset();
    }
}

bool CreateGraphicsPipeline(VkDevice device, const vk::DeviceDispatch& dld,
                            const VkGraphicsPipelineCreateInfo& ci, VkPipelineCache cache,
                            VkPipeline* out, VkResult* result) {
    std::shared_lock lock{active_mutex};
    if (!active) {
        return false;
    }
    return active->Create(ci, cache, out, result,
                          [&](const VkGraphicsPipelineCreateInfo& info, VkPipelineCache c,
                              VkPipeline* p) {
                              return dld.vkCreateGraphicsPipelines(device, c, 1, &info, nullptr,
                                                                   p);
                          });
}

bool CreateComputePipeline(VkDevice device, const vk::DeviceDispatch& dld,
                           const VkComputePipelineCreateInfo& ci, VkPipelineCache cache,
                           VkPipeline* out, VkResult* result) {
    std::shared_lock lock{active_mutex};
    if (!active) {
        return false;
    }
    return active->Create(ci, cache, out, result,
                          [&](const VkComputePipelineCreateInfo& info, VkPipelineCache c,
                              VkPipeline* p) {
                              return dld.vkCreateComputePipelines(device, c, 1, &info, nullptr,
                                                                  p);
                          });
}

} // namespace Vulkan::PipelineBinaryCache
