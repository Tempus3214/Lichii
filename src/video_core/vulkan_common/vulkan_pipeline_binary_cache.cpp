// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
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
// A later record for a key replaces an earlier one: a binary the driver rejects is saved again.
constexpr std::array<char, 8> FILE_MAGIC{'L', 'C', 'H', 'P', 'B', 'I', 'N', '1'};
constexpr u32 RECORD_MAGIC = 0x44524350;
constexpr size_t KEY_BYTES = VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;
constexpr size_t HEADER_BYTES = FILE_MAGIC.size() + 4 + KEY_BYTES;
constexpr size_t RECORD_HEADER_BYTES = 4 + 4 + KEY_BYTES + 8;
constexpr u32 MAX_BINARIES_PER_PIPELINE = 64;

// The file is rewritten when it closes if replaced records take at least MIN_DEAD_BYTES and a
// quarter of it, or if it is larger than MAX_FILE_BYTES. Past that size the pipelines used in the
// session are kept first, then the newest others, up to three quarters of it.
constexpr u64 MAX_FILE_BYTES = 512ULL << 20;
constexpr u64 MIN_DEAD_BYTES = 64ULL << 20;

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
        std::memcpy(header.data(), FILE_MAGIC.data(), FILE_MAGIC.size());
        std::memcpy(header.data() + 8, &global.keySize, 4);
        std::memcpy(header.data() + 12, global.key, global.keySize);

        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        u64 size = std::filesystem::file_size(path, ec);
        if (ec) {
            size = 0;
        }
        bool same_driver = false;
        {
            std::ifstream scan(path, std::ios::binary);
            std::array<u8, HEADER_BYTES> found{};
            same_driver = scan && size >= HEADER_BYTES &&
                          scan.read(reinterpret_cast<char*>(found.data()), HEADER_BYTES) &&
                          found == header;
            if (same_driver) {
                file_end = HEADER_BYTES;
                std::array<u8, RECORD_HEADER_BYTES> record{};
                while (file_end + RECORD_HEADER_BYTES <= size &&
                       scan.read(reinterpret_cast<char*>(record.data()), RECORD_HEADER_BYTES)) {
                    u32 magic, key_size;
                    u64 len;
                    std::memcpy(&magic, &record[0], 4);
                    std::memcpy(&key_size, &record[4], 4);
                    std::memcpy(&len, &record[8 + KEY_BYTES], 8);
                    if (magic != RECORD_MAGIC || key_size == 0 || key_size > KEY_BYTES ||
                        len > size - file_end - RECORD_HEADER_BYTES) {
                        break;
                    }
                    const u64 payload = file_end + RECORD_HEADER_BYTES;
                    Remember(std::string(reinterpret_cast<const char*>(&record[8]), key_size),
                             {payload, len, false});
                    file_end = payload + len;
                    scan.seekg(static_cast<std::streamoff>(file_end));
                }
            }
        }
        if (same_driver) {
            if (file_end < size) {
                // A session that ended mid-write leaves a partial record; drop it.
                std::filesystem::resize_file(path, file_end, ec);
            }
            out.open(path, std::ios::binary | std::ios::app);
        } else {
            // First run, or the driver changed: binaries from another driver can't be used, so the
            // file is replaced rather than kept beside the new one.
            if (size != 0) {
                LOG_INFO(Render_Vulkan, "Pipeline binaries: driver changed, starting over");
            }
            index.clear();
            dead_bytes = 0;
            out.open(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(header.data()), header.size());
            out.flush();
            file_end = HEADER_BYTES;
        }
        in.open(path, std::ios::binary);
        if (!out || !in) {
            LOG_ERROR(Render_Vulkan, "Pipeline binaries: can't open {}", path.string());
            return false;
        }
        LOG_INFO(Render_Vulkan,
                 "Pipeline binaries: {} saved pipelines ({:.1f} MiB, {:.1f} MiB replaced) in {}",
                 index.size(), Mib(file_end), Mib(dead_bytes), path.string());
        return true;
    }

    void Close() {
        std::scoped_lock lock{mutex};
        out.close();
        Compact();
        in.close();
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

        if (const auto payload = Read(key_bytes)) {
            if (CreateFromSaved(*payload, ci, out_pipeline, create)) {
                MarkUsed(key_bytes);
                ++stats.loaded;
                stats.loaded_us += elapsed_us();
                *result = VK_SUCCESS;
                return true;
            }
            // Compiled and saved again below, replacing the rejected record.
            Forget(key_bytes);
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
    struct Record {
        u64 offset;
        u64 size;
        bool used;
    };

    static double Mib(u64 bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

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

    // Callers hold the mutex, or own the store alone (Open).
    void Remember(const std::string& key, const Record& record) {
        const auto [it, inserted] = index.try_emplace(key, record);
        if (!inserted) {
            dead_bytes += RECORD_HEADER_BYTES + it->second.size;
            it->second = record;
        }
    }

    std::optional<std::vector<u8>> Read(const std::string& key) {
        std::scoped_lock lock{mutex};
        const auto it = index.find(key);
        if (it == index.end()) {
            return std::nullopt;
        }
        std::vector<u8> payload(static_cast<size_t>(it->second.size));
        in.clear();
        in.seekg(static_cast<std::streamoff>(it->second.offset));
        if (!in.read(reinterpret_cast<char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size()))) {
            return std::nullopt;
        }
        return payload;
    }

    void MarkUsed(const std::string& key) {
        std::scoped_lock lock{mutex};
        if (const auto it = index.find(key); it != index.end()) {
            it->second.used = true;
        }
    }

    void Forget(const std::string& key) {
        std::scoped_lock lock{mutex};
        if (const auto it = index.find(key); it != index.end()) {
            dead_bytes += RECORD_HEADER_BYTES + it->second.size;
            index.erase(it);
        }
    }

    template <typename CreateInfo, typename CreateFn>
    bool CreateFromSaved(const std::vector<u8>& blob, const CreateInfo& ci,
                         VkPipeline* out_pipeline, CreateFn& create) {
        if (blob.size() < 4) {
            return false;
        }
        u32 count;
        std::memcpy(&count, blob.data(), 4);
        if (count == 0 || count > MAX_BINARIES_PER_PIPELINE) {
            return false;
        }
        std::vector<VkPipelineBinaryKeyKHR> keys(count);
        std::vector<VkPipelineBinaryDataKHR> datas(count);
        size_t o = 4;
        for (u32 i = 0; i < count; ++i) {
            if (o + 4 + KEY_BYTES + 8 > blob.size()) {
                return false;
            }
            keys[i] = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            std::memcpy(&keys[i].keySize, blob.data() + o, 4);
            if (keys[i].keySize > KEY_BYTES) {
                return false;
            }
            std::memcpy(keys[i].key, blob.data() + o + 4, KEY_BYTES);
            u64 size;
            std::memcpy(&size, blob.data() + o + 4 + KEY_BYTES, 8);
            o += 4 + KEY_BYTES + 8;
            if (size > blob.size() - o) {
                return false;
            }
            datas[i] = {.dataSize = static_cast<size_t>(size),
                        .pData = const_cast<u8*>(blob.data() + o)};
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
        std::scoped_lock lock{mutex};
        if (index.contains(key) || !out.is_open()) {
            return;
        }
        WriteRecord(out, key, payload.data(), payload.size());
        out.flush();
        if (!out) {
            LOG_ERROR(Render_Vulkan, "Pipeline binaries: can't write {}, saving stopped",
                      path.string());
            out.close();
            return;
        }
        index.emplace(key, Record{file_end + RECORD_HEADER_BYTES, payload.size(), true});
        file_end += RECORD_HEADER_BYTES + payload.size();
        ++stats.saved;
    }

    static void WriteRecord(std::ofstream& file, const std::string& key, const u8* data,
                            u64 size) {
        const u32 magic = RECORD_MAGIC;
        const u32 key_size = static_cast<u32>(key.size());
        std::array<u8, KEY_BYTES> key_padded{};
        std::memcpy(key_padded.data(), key.data(), key.size());
        file.write(reinterpret_cast<const char*>(&magic), 4);
        file.write(reinterpret_cast<const char*>(&key_size), 4);
        file.write(reinterpret_cast<const char*>(key_padded.data()), KEY_BYTES);
        file.write(reinterpret_cast<const char*>(&size), 8);
        file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    // Called with the mutex held and the append stream closed.
    void Compact() {
        const bool too_large = file_end > MAX_FILE_BYTES;
        if (!too_large && (dead_bytes < MIN_DEAD_BYTES || dead_bytes * 4 < file_end)) {
            return;
        }
        std::vector<std::pair<const std::string*, const Record*>> keep;
        keep.reserve(index.size());
        for (const auto& [key, record] : index) {
            keep.emplace_back(&key, &record);
        }
        // Used this session first, then newest first; the file keeps its record order.
        std::ranges::sort(keep, [](const auto& a, const auto& b) {
            return a.second->used != b.second->used ? a.second->used
                                                    : a.second->offset > b.second->offset;
        });
        u64 kept_bytes = HEADER_BYTES;
        size_t kept = 0;
        const u64 budget = too_large ? MAX_FILE_BYTES / 4 * 3 : ~0ULL;
        while (kept < keep.size() &&
               kept_bytes + RECORD_HEADER_BYTES + keep[kept].second->size <= budget) {
            kept_bytes += RECORD_HEADER_BYTES + keep[kept].second->size;
            ++kept;
        }
        keep.resize(kept);
        std::ranges::sort(keep, {}, [](const auto& entry) { return entry.second->offset; });

        auto temp = path;
        temp += ".tmp";
        bool ok = true;
        {
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(header.data()), header.size());
            std::vector<u8> payload;
            for (const auto& [key, record] : keep) {
                payload.resize(static_cast<size_t>(record->size));
                in.clear();
                in.seekg(static_cast<std::streamoff>(record->offset));
                if (!in.read(reinterpret_cast<char*>(payload.data()),
                             static_cast<std::streamsize>(payload.size()))) {
                    ok = false;
                    break;
                }
                WriteRecord(file, *key, payload.data(), payload.size());
            }
            file.flush();
            ok = ok && file.good();
        }
        in.close();
        std::error_code ec;
        if (ok) {
            std::filesystem::rename(temp, path, ec);
        }
        if (!ok || ec) {
            std::filesystem::remove(temp, ec);
            LOG_WARNING(Render_Vulkan, "Pipeline binaries: can't rewrite {}, keeping it",
                        path.string());
            return;
        }
        LOG_INFO(Render_Vulkan,
                 "Pipeline binaries: rewrote {} with {} of {} pipelines ({:.1f} of {:.1f} MiB)",
                 path.string(), keep.size(), index.size(), Mib(kept_bytes), Mib(file_end));
    }

    VkDevice device{};
    const vk::DeviceDispatch* dld{};
    std::filesystem::path path;
    std::array<u8, HEADER_BYTES> header{};
    std::ofstream out;
    std::ifstream in;
    u64 file_end{};
    u64 dead_bytes{};
    std::unordered_map<std::string, Record> index;
    std::mutex mutex;
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
