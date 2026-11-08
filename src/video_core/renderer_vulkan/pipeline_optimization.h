// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

namespace Vulkan {

inline bool BuildsUnoptimized(VkDriverId driver_id, u64 shader_hash) {
    // The Legend of Zelda: Ocarina of Time (0400486024DEC000), driver 2.0.353 on an RX 6600 XT
    static constexpr std::array<u64, 3> amd_proprietary{
        0xeff3ccbe1438679dULL, // compute shader built as the AV1 intro starts: 6 min (#200)
        0xf9ea14a22394f5b1ULL, // compute shader built as the AV1 intro starts: 18 min (#200)
        0xe0a992208d87eeb8ULL, // Kokiri Forest compute shader: 1.3-4 s stalls (RX 9070 XT, 2.0.416)
    };
    return driver_id == VK_DRIVER_ID_AMD_PROPRIETARY &&
           std::ranges::find(amd_proprietary, shader_hash) != amd_proprietary.end();
}

} // namespace Vulkan
