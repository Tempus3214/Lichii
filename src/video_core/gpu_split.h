// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA)
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",       "fetch drain",     "acquire",    "semaphore flush", "fence release",
    "list writers", "list sync",      "list flush", "macro dma read",  "macro dma flush",
    "gp sync fence", "map",           "front start", "front verify",
};

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    template <typename Func>
    void Defer(const Func& func) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload);
    }
    void DeferFunction(std::function<void()>&& func);
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// Compute or copy work was just sent: what can write GPU-written command lists.
    void NoteWriter() {
        last_writer_position = write_position;
    }
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires drain): the guest's GPU work before it is complete before the
    /// commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
        NoteIdle();
    }
    /// A draw was just sent. Draws write memory the front does not track (shader stores).
    void NoteDraw() {
        draws_queued = true;
    }
    /// A wait for idle was just sent: what the draws before it wrote is visible after it, as
    /// the writes of compute work are.
    void NoteIdle() {
        if (draws_queued) {
            draws_queued = false;
            last_writer_position = write_position;
        }
    }
    /// Increments each time the split stops: ring positions recorded before then are void.
    [[nodiscard]] u64 Epoch() const {
        return epoch;
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();

    /// Work that may write guest memory at addresses the front does not track was just sent.
    /// `source` names the sender for EDEN_NX2_MME_DMA_STATS.
    void NoteMemoryBarrier(u32 source = 0) {
        last_barrier_position = write_position;
        last_barrier_source = source;
    }
    [[nodiscard]] u32 MemoryBarrierSource() const {
        return last_barrier_source;
    }
    /// Where the last untracked memory write (or compute and copy work) was sent.
    [[nodiscard]] u64 MemoryBarrierPosition() const {
        return std::max(last_barrier_position, last_writer_position);
    }
    [[nodiscard]] u64 WriterPosition() const {
        return last_writer_position;
    }
    [[nodiscard]] u64 Position() const {
        return write_position;
    }
    /// How far the back has executed.
    [[nodiscard]] u64 Consumed() {
        consumed_cache = consumed.load(std::memory_order_acquire);
        return consumed_cache;
    }
    /// Waits until the back executed everything sent before `target`.
    void WaitForPosition(u64 target, DrainSite site) {
        WaitFor(std::min(target, write_position), site);
    }

    /// Work just sent writes [begin, end) of the GPU address space `space` (a memory manager).
    /// `opaque`: the front does not know the data written.
    void NoteMemoryWrite(const void* space, u64 begin, u64 end, bool opaque = true) {
        if (end <= begin) {
            return;
        }
        if (((end - 1) >> WriteBlockShift) - (begin >> WriteBlockShift) >= MaxWriteBlocks) {
            NoteMemoryBarrier(0x40001);
            return;
        }
        if (write_blocks.size() >= 8192 && Consumed() >= latest_block_write) {
            write_blocks.clear();
            opaque_blocks.clear();
        }
        if (write_blocks.size() >= 65536) {
            write_blocks.clear();
            opaque_blocks.clear();
            NoteMemoryBarrier(0x40000);
        }
        NoteBlocks(write_blocks, space, begin, end);
        latest_block_write = write_position;
        if (opaque) {
            NoteBlocks(opaque_blocks, space, begin, end);
            latest_opaque_write = write_position;
        }
    }
    /// The position the back must reach before [begin, end) of `space` can be read: the last
    /// untracked write, or the last queued write that overlaps it.
    [[nodiscard]] u64 ReadReadyPosition(const void* space, u64 begin, u64 end) {
        return ReadyPosition(write_blocks, latest_block_write, space, begin, end);
    }
    /// As ReadReadyPosition, for writes whose data the front does not know.
    [[nodiscard]] u64 OpaqueReadyPosition(const void* space, u64 begin, u64 end) {
        return ReadyPosition(opaque_blocks, latest_opaque_write, space, begin, end);
    }
    /// The position after the last queued tracked write to [begin, end), untracked writes aside.
    [[nodiscard]] u64 TrackedWritePosition(const void* space, u64 begin, u64 end);

    /// Front only: a constant buffer word sent inline and the ring position after its upload.
    struct FrontWord {
        u32 value;
        u64 position;
    };
    /// Front only: the words of every channel, keyed by device address, so the last upload of a
    /// word wins whichever channel sent it.
    [[nodiscard]] std::unordered_map<u64, FrontWord>& FrontWords() {
        return front_words;
    }

private:
    friend struct GpuSplitTestAccess;

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words) {
        write_position += words;
        if (write_position - published_local >= publish_every) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);

    Core::System* system{};
    bool active{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_barrier_position{};
    bool draws_queued{};
    u64 epoch{};
    u32 last_barrier_source{};
    // Queued tracked writes: the ring position after the last write to each 64-byte block.
    static constexpr u64 WriteBlockShift = 6;
    static constexpr u64 MaxWriteBlocks = 1024;
    std::unordered_map<u64, u64> write_blocks;
    std::unordered_map<u64, u64> opaque_blocks;
    std::unordered_map<u64, FrontWord> front_words;
    u64 latest_block_write{};
    u64 latest_opaque_write{};
    // Keyed by the backing memory (`space` is a Tegra::MemoryManager), so writes and reads
    // through different GPU addresses of the same memory meet.
    void NoteBlocks(std::unordered_map<u64, u64>& blocks, const void* space, u64 begin, u64 end);
    u64 BlocksPosition(const std::unordered_map<u64, u64>& blocks, const void* space, u64 begin,
                       u64 end) const;
    u64 ReadyPosition(const std::unordered_map<u64, u64>& blocks, u64 latest, const void* space,
                      u64 begin, u64 end) {
        u64 target = MemoryBarrierPosition();
        if (end <= begin || blocks.empty() || latest <= target) {
            return target;
        }
        if (((end - 1) >> WriteBlockShift) - (begin >> WriteBlockShift) >= MaxWriteBlocks) {
            return std::max(target, latest);
        }
        return std::max(target, BlocksPosition(blocks, space, begin, end));
    }
    u64 last_sync_position{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    std::jthread back_thread;
};

} // namespace VideoCommon
