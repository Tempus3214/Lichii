// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/glasm/emit_glasm.h"
#include "shader_recompiler/backend/glsl/emit_glsl.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
using namespace Shader;

class TestEnvironment final : public Environment {
public:
    explicit TestEnvironment(bool native_ = false, std::span<const u64> code_ = {})
        : native{native_}, code{code_} {
        stage = Stage::Compute;
    }
    u64 ReadInstruction(u32 address) override {
        if (code.empty()) {
            return 0;
        }
        for (u32 index = 0; index < code.size(); ++index) {
            if (address == 0x50 + index * 8 + ((index + 1) / 3) * 8) {
                return code[index];
            }
        }
        FAIL("Read outside shader fixture at " << address);
        return 0;
    }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    TextureType ReadTextureType(u32) override { return TextureType::Color2D; }
    TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    bool HasNativeInstructions() const override { return native; }
    bool IsNativeProgram() const override { return native; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}
    bool native;
    std::span<const u64> code;
};

struct Program {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile{};

    Program() {
        program.stage = Stage::Compute;
        program.workgroup_size = {32, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        profile.supported_spirv = 0x00010300;
        profile.min_ssbo_alignment = 16;
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t i = 5; i < code.size();) {
        const u32 count = code[i] >> 16;
        REQUIRE(count > 0);
        REQUIRE(i + count <= code.size());
        if ((code[i] & 0xffff) == opcode) {
            result.emplace_back(code.begin() + i + 1, code.begin() + i + count);
        }
        i += count;
    }
    return result;
}

struct GlobalAtomicProgram : Program {
    explicit GlobalAtomicProgram(bool native, s64 displacement = 4, bool atomic = true) {
        TestEnvironment env{native};
        profile.support_int64 = true;
        profile.support_descriptor_aliasing = true;
        profile.unified_descriptor_binding = true;
        IR::IREmitter ir{block};
        const auto word = [&](u32 bank, u32 offset) {
            return ir.GetCbuf(ir.Imm32(bank), ir.Imm32(offset));
        };
        const u32 first = native ? 0x168 : 0x340;
        const u32 second = native ? 0x178 : 0x400;
        const auto lane = ir.LocalInvocationIdX();
        const auto select_first = ir.IEqual(ir.BitwiseAnd(lane, ir.Imm32(1)), ir.Imm32(0));
        const auto low = ir.Select(select_first, word(0, first), word(0, second));
        const auto high = ir.Select(select_first, word(0, first + 4), word(0, second + 4));
        const IR::U64 address{ir.IAdd(ir.PackUint2x32(ir.CompositeConstruct(low, high)),
                                     ir.Imm64(displacement))};
        const IR::U32 previous{atomic ? IR::U32{ir.GlobalAtomicIAdd(address, ir.Imm32(1))}
                                     : ir.LoadGlobal32(address)};
        const IR::U32 index{ir.IAdd(ir.IMul(ir.WorkgroupIdX(), ir.Imm32(32)), lane)};
        const IR::U32 offset{ir.ShiftLeftLogical(index, ir.Imm32(2))};
        if (native) {
            const auto output = ir.PackUint2x32(ir.CompositeConstruct(
                ir.IAdd(word(1, 0), offset), word(1, 4)));
            ir.WriteGlobal32(output, previous);
            HostTranslateInfo host{};
            host.min_ssbo_alignment = 16;
            Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
        } else {
            block.AppendNewInst(IR::Opcode::WriteStorage32, {ir.Imm32(0), offset, previous});
            program.info.storage_buffers_descriptors = {
                {1, 0, 1, true}, {0, first, 1, true}, {0, second, 1, true},
            };
        }
        Optimization::CollectShaderInfoPass(env, program);
        program.info.constant_buffer_descriptors = {{0, 1}, {1, 1}};
    }
};
} // namespace

TEST_CASE("Native LDG cache policy reaches every global load width", "[video_core][coherence]") {
    for (const bool native : {false, true}) {
        for (u32 cache = 0; cache < 4; ++cache) {
            for (u32 size = 0; size < 7; ++size) {
                TestEnvironment env{native};
                Program fixture;
                Maxwell::TranslatorVisitor visitor{env, fixture.block};
                visitor.LDG((u64{cache} << 46) | (u64{size} << 48));
                const auto& instructions = fixture.block.Instructions();
                const auto load = std::ranges::find_if(instructions, [](const IR::Inst& inst) {
                    return inst.GetOpcode() >= IR::Opcode::LoadGlobalU8 &&
                           inst.GetOpcode() <= IR::Opcode::LoadGlobal128;
                });
                REQUIRE(load != instructions.end());
                CHECK(load->Flags<bool>() == (native && cache == 1));
                Optimization::CollectShaderInfoPass(env, fixture.program);
                CHECK(fixture.program.info.requires_coherent_global_memory ==
                      (native && cache == 1));
            }
        }
    }
}

TEST_CASE("Global load coherency survives storage buffer tracking", "[video_core][coherence]") {
    for (const bool tracked : {false, true}) {
        for (const bool coherent : {false, true}) {
            for (u32 size = 0; size < 7; ++size) {
                TestEnvironment env;
                Program fixture;
                IR::IREmitter ir{fixture.block};
                const auto low = tracked ? ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x310))
                                         : ir.LocalInvocationIdX();
                const auto address = ir.PackUint2x32(ir.CompositeConstruct(low, ir.Imm32(0)));
                switch (size) {
                case 0: (void)ir.LoadGlobalU8(address, coherent); break;
                case 1: (void)ir.LoadGlobalS8(address, coherent); break;
                case 2: (void)ir.LoadGlobalU16(address, coherent); break;
                case 3: (void)ir.LoadGlobalS16(address, coherent); break;
                case 4: (void)ir.LoadGlobal32(address, coherent); break;
                case 5: (void)ir.LoadGlobal64(address, coherent); break;
                case 6: (void)ir.LoadGlobal128(address, coherent); break;
                }
                HostTranslateInfo host{};
                host.min_ssbo_alignment = 16;
                Optimization::GlobalMemoryToStorageBufferPass(env, fixture.program, host);
                const auto expected = static_cast<IR::Opcode>(
                    static_cast<u32>(tracked ? IR::Opcode::LoadStorageU8 : IR::Opcode::LoadGlobalU8) + size);
                const auto& instructions = fixture.block.Instructions();
                const auto load = std::ranges::find_if(instructions, [expected](const IR::Inst& inst) {
                    return inst.GetOpcode() == expected;
                });
                REQUIRE(load != instructions.end());
                CHECK(load->Flags<bool>() == coherent);
                Optimization::CollectShaderInfoPass(env, fixture.program);
                CHECK(fixture.program.info.requires_coherent_global_memory == coherent);
            }
        }
    }
}

TEST_CASE("Coherent storage aliases and barrier semantics are emitted in SPIR-V",
          "[video_core][coherence]") {
    for (const bool coherent : {false, true}) {
        for (const bool aliasing : {false, true}) {
            Program fixture;
            fixture.profile.support_descriptor_aliasing = aliasing;
            fixture.program.info.requires_coherent_global_memory = coherent;
            fixture.program.info.used_storage_buffer_types = IR::Type::U32 | IR::Type::F32;
            fixture.program.info.storage_buffers_descriptors = {{0, 0, 1, true}, {0, 16, 1, true}};
            IR::IREmitter ir{fixture.block};
            ir.Barrier();
            const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
            std::set<u32> blocks, coherent_members, storage_structs;
            std::map<u32, u32> pointers, constants;
            for (const auto& op : Instructions(code, 71)) {
                if (op.size() >= 2 && op[1] == 2) blocks.insert(op[0]);
            }
            for (const auto& op : Instructions(code, 72)) {
                if (op.size() >= 3 && op[1] == 0 && op[2] == 23) coherent_members.insert(op[0]);
            }
            for (const auto& op : Instructions(code, 32)) {
                if (op.size() >= 3 && op[1] == 12) pointers[op[0]] = op[2];
            }
            for (const auto& op : Instructions(code, 59)) {
                if (op.size() >= 3 && op[2] == 12) {
                    REQUIRE(pointers.contains(op[0]));
                    storage_structs.insert(pointers.at(op[0]));
                }
            }
            REQUIRE(storage_structs.size() == (aliasing ? 2U : 1U));
            for (const u32 type : storage_structs) {
                CHECK(blocks.contains(type));
                CHECK(coherent_members.contains(type) == coherent);
            }
            for (const auto& op : Instructions(code, 43)) {
                if (op.size() >= 3) constants[op[1]] = op[2];
            }
            const auto barriers = Instructions(code, 224);
            REQUIRE(barriers.size() == 1);
            REQUIRE(barriers[0].size() == 3);
            REQUIRE(constants.contains(barriers[0][2]));
            CHECK(constants.at(barriers[0][2]) == (0x108U | (coherent ? 0x40U : 0U)));
        }
    }
}

TEST_CASE("GLSL and GLASM emit coherent storage and barrier operations",
          "[video_core][coherence]") {
    for (const bool coherent : {false, true}) {
        for (const bool pointer : {false, true}) {
            for (const bool fallback : {false, true}) {
              for (const bool glsl_backend : {false, true}) {
                Program fixture;
                auto& info = fixture.program.info;
                info.requires_coherent_global_memory = coherent;
                info.used_storage_buffer_types = IR::Type::U32;
                info.storage_buffers_descriptors = {{0, 0, 1, true}};
                info.nvn_buffer_used.set(0);
                info.uses_global_memory = fallback;
                info.constant_buffer_descriptors = {{0, 1}};
                fixture.profile.support_int64 = true;
                IR::IREmitter ir{fixture.block};
                const auto offset = ir.LocalInvocationIdX();
                info.uses_local_invocation_id = true;
                if (fallback) {
                    const auto addr = ir.PackUint2x32(ir.CompositeConstruct(offset, ir.Imm32(0)));
                    const auto value = ir.LoadGlobal32(addr, coherent);
                    ir.WriteGlobal32(addr, value);
                } else {
                    fixture.block.AppendNewInst(IR::Opcode::LoadStorage32, {ir.Imm32(0), offset});
                    fixture.block.AppendNewInst(IR::Opcode::WriteStorage32,
                                                {ir.Imm32(0), offset, ir.Imm32(42)});
                }
                ir.Barrier();
                if (glsl_backend) {
                    const auto glsl = Backend::GLSL::EmitGLSL(fixture.profile, fixture.program);
                    CHECK((glsl.find("coherent ") != std::string::npos) == coherent);
                    CHECK((glsl.find("groupMemoryBarrier();") != std::string::npos) == coherent);
                    CHECK(glsl.find("barrier();") != std::string::npos);
                    if (fallback) {
                        CHECK(glsl.find("uint LoadGlobal32(uint64_t addr)") != std::string::npos);
                        CHECK(glsl.find("void WriteGlobal32(uint64_t addr,uint data)") != std::string::npos);
                    }
                } else {
                    RuntimeInfo runtime{};
                    runtime.glasm_use_storage_buffers = !pointer;
                    const auto code = Backend::GLASM::EmitGLASM(fixture.profile, runtime, fixture.program);
                    const std::string load = pointer ? "LOAD" : "LDB";
                    const std::string store = pointer ? "STORE" : "STB";
                    CHECK(code.find(load + (coherent ? ".COH.U32" : ".U32")) != std::string::npos);
                    CHECK(code.find(store + (coherent ? ".COH.U32" : ".U32")) != std::string::npos);
                    CHECK((code.find("MEMBAR.CTA;\nBAR;") != std::string::npos) == coherent);
                }
              }
            }
        }
    }
}

TEST_CASE("Joining vertex storage preserves coherent memory requirement", "[video_core][coherence]") {
    for (const bool a : {false, true}) {
        for (const bool b : {false, true}) {
            Info vertex_a{}, vertex_b{};
            vertex_a.requires_coherent_global_memory = a;
            vertex_b.requires_coherent_global_memory = b;
            Optimization::JoinStorageInfo(vertex_a, vertex_b);
            CHECK(vertex_a.requires_coherent_global_memory == (a || b));
        }
    }
}

TEST_CASE("Global fallback backends address sparse guest slots through compact bindings",
          "[shader][storage_buffer][nvn]") {
    for (const u32 backend : {0U, 1U, 2U}) {
        CAPTURE(backend);
        Program fixture;
        auto& info = fixture.program.info;
        info.used_storage_buffer_types = IR::Type::U32 | IR::Type::U32x2 | IR::Type::U32x4;
        info.storage_buffers_descriptors = {
            {1, 0x340, 1, true}, {0, 0x340, 1, true},
            {0, 0x400, 1, true}, {0, 0x330, 1, true},
        };
        info.nvn_buffer_base = 0x310;
        info.nvn_buffer_used.set(3);
        info.nvn_buffer_used.set(15);
        info.uses_global_memory = true;
        info.uses_int64 = true;
        info.used_constant_buffer_types = IR::Type::U32 | IR::Type::U32x2;
        info.stores_global_memory = true;
        info.constant_buffer_descriptors = {{0, 1}, {1, 1}};
        info.uses_local_invocation_id = true;
        fixture.profile.support_int64 = true;
        fixture.profile.support_descriptor_aliasing = true;
        IR::IREmitter ir{fixture.block};
        const auto address = ir.PackUint2x32(
            ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.Imm32(0)));
        ir.WriteGlobal32(address, ir.Imm32(42));

        if (backend == 0) {
            const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
            std::set<u32> storage_variables;
            std::map<u32, u32> bindings;
            for (const auto& op : Instructions(code, 59)) {
                if (op.size() >= 3 && op[2] == 12) {
                    storage_variables.insert(op[1]);
                }
            }
            for (const auto& op : Instructions(code, 71)) {
                if (op.size() >= 3 && op[1] == 33) {
                    bindings[op[0]] = op[2];
                }
            }
            std::set<u32> accessed_bindings;
            for (const auto& op : Instructions(code, 65)) {
                if (op.size() >= 3 && storage_variables.contains(op[2])) {
                    REQUIRE(bindings.contains(op[2]));
                    accessed_bindings.insert(bindings.at(op[2]));
                }
            }
            std::set<u32> declared_bindings;
            for (const u32 variable : storage_variables) {
                REQUIRE(bindings.contains(variable));
                declared_bindings.insert(bindings.at(variable));
            }
            REQUIRE(declared_bindings.size() == 4);
            std::set<u32> expected_bindings;
            const u32 first_storage_binding = *declared_bindings.begin();
            for (const u32 binding : {1U, 2U}) {
                expected_bindings.insert(first_storage_binding + binding);
            }
            CHECK(accessed_bindings == expected_bindings);
        } else if (backend == 1) {
            const auto code = Backend::GLSL::EmitGLSL(fixture.profile, fixture.program);
            CHECK(code.find("uint64_t ssbo_addr1=") != std::string::npos);
            CHECK(code.find("uint64_t ssbo_addr2=") != std::string::npos);
            CHECK(code.find("uint64_t ssbo_addr0=") == std::string::npos);
            CHECK(code.find("uint64_t ssbo_addr3=") == std::string::npos);
        } else {
            RuntimeInfo runtime{};
            runtime.glasm_use_storage_buffers = true;
            const auto code = Backend::GLASM::EmitGLASM(fixture.profile, runtime, fixture.program);
            CHECK(code.find("LDC.U64 DC.x,c0[832];") != std::string::npos);
            CHECK(code.find("LDC.U64 DC.x,c0[1024];") != std::string::npos);
            CHECK(code.find("LDC.U64 DC.x,c1[832];") == std::string::npos);
            CHECK(code.find("LDC.U64 DC.x,c0[816];") == std::string::npos);
            CHECK(code.find("ssbo1[RC.x]") != std::string::npos);
            CHECK(code.find("ssbo2[RC.x]") != std::string::npos);
        }
    }
}

TEST_CASE("SPIR-V global memory helpers are DontInline in compute only",
          "[video_core][spirv][global_memory]") {
    for (const Stage stage : {Stage::Compute, Stage::VertexB, Stage::Fragment, Stage::Mesh}) {
        CAPTURE(stage);
        Program fixture;
        fixture.program.stage = stage;
        if (stage == Stage::Mesh) {
            fixture.program.mesh.max_vertices = 32;
            fixture.program.mesh.max_primitives = 1;
            fixture.program.mesh.topology = 4;
        }
        auto& info = fixture.program.info;
        info.used_storage_buffer_types = IR::Type::U32;
        info.storage_buffers_descriptors = {{0, 0x340, 1, true}, {0, 0x400, 1, true}};
        info.uses_global_memory = true;
        info.uses_int64 = true;
        info.stores_global_memory = true;
        info.constant_buffer_descriptors = {{0, 1}};
        info.used_constant_buffer_types = IR::Type::U32 | IR::Type::U32x2;
        fixture.profile.support_int64 = true;
        fixture.profile.support_descriptor_aliasing = true;
        IR::IREmitter ir{fixture.block};
        const auto address = ir.PackUint2x32(
            ir.CompositeConstruct(ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x10)), ir.Imm32(0)));
        ir.WriteGlobal32(address, ir.LoadGlobal32(address));

        const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
        constexpr u32 OpEntryPoint = 15;
        constexpr u32 OpFunction = 54;
        constexpr u32 OpFunctionCall = 57;
        constexpr u32 DontInline = 0x2;
        std::map<u32, u32> controls;
        for (const auto& op : Instructions(code, OpFunction)) {
            REQUIRE(op.size() == 4);
            controls[op[1]] = op[2];
        }
        const auto entry_points = Instructions(code, OpEntryPoint);
        REQUIRE(entry_points.size() == 1);
        REQUIRE(controls.contains(entry_points[0][1]));
        CHECK((controls.at(entry_points[0][1]) & DontInline) == 0);

        std::set<u32> callees;
        for (const auto& op : Instructions(code, OpFunctionCall)) {
            REQUIRE(op.size() >= 3);
            callees.insert(op[2]);
        }
        REQUIRE(callees.size() == 2);
        const u32 expected = stage == Stage::Compute ? DontInline : 0;
        for (const u32 callee : callees) {
            CAPTURE(callee);
            REQUIRE(controls.contains(callee));
            CHECK((controls.at(callee) & DontInline) == expected);
        }
    }
}

TEST_CASE("NVN fallback does not deduplicate descriptors from different constant buffers",
          "[shader][storage_buffer][nvn]") {
    const std::array<u64, 5> code{
        0xef90000000070000ULL | (u64{255} << 8) | (u64{0x310} << 20) |
            (u64{1} << 36) | (u64{5} << 48),
        0xeed0000000070000ULL | 2 | (u64{1} << 45) | (u64{5} << 48),
        0xef90000000070000ULL | 4 | (u64{255} << 8) | (u64{0x310} << 20) |
            (u64{4} << 48),
        0xeed8000000070000ULL | 4 | (u64{2} << 8) | (u64{1} << 45) | (u64{4} << 48),
        0xe30000000007000fULL,
    };
    TestEnvironment env{false, code};
    ObjectPool<Maxwell::Flow::Block> flow_pool;
    Maxwell::Flow::CFG cfg{env, flow_pool, Maxwell::Location{0x50}};
    ObjectPool<IR::Inst> inst_pool;
    ObjectPool<IR::Block> block_pool;
    HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    host.support_int64 = true;
    auto program = Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    REQUIRE(program.info.uses_global_memory);
    REQUIRE(program.info.nvn_buffer_used.test(0));
    const auto& descriptors = program.info.storage_buffers_descriptors;
    REQUIRE(descriptors.size() == 2);
    CHECK(descriptors[0].cbuf_index == 1);
    CHECK(descriptors[0].cbuf_offset == 0x310);
    CHECK_FALSE(descriptors[0].is_written);
    CHECK(descriptors[1].cbuf_index == 0);
    CHECK(descriptors[1].cbuf_offset == 0x310);
    CHECK(descriptors[1].is_written);
}

TEST_CASE("SPIR-V emission failures identify the unsupported IR opcode",
          "[shader][spirv][diagnostics]") {
    for (const IR::Opcode opcode : {IR::Opcode::UndefU8, IR::Opcode::UndefU16}) {
        Program fixture;
        fixture.block.AppendNewInst(opcode, {});
        try {
            static_cast<void>(Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program));
            FAIL("Unsupported IR instruction unexpectedly emitted");
        } catch (const Shader::NotImplementedException& exception) {
            const std::string message{exception.what()};
            CHECK(message.find(IR::NameOf(opcode)) != std::string::npos);
            CHECK(message.find("SPIR-V Instruction is not implemented") != std::string::npos);
        }
    }
}

TEST_CASE("Global atomic add preserves native candidate selection and atomic old values",
          "[shader][spirv][global_atomic]") {
    for (const bool native : {true, false}) {
        CAPTURE(native);
        GlobalAtomicProgram fixture{native};
        const auto& info = fixture.program.info;
        REQUIRE(info.native_global_bounds == native);
        REQUIRE(info.storage_buffers_descriptors.size() == 3);
        CHECK(std::ranges::all_of(info.storage_buffers_descriptors,
                                  [](const auto& descriptor) { return descriptor.is_written; }));
        if (native) {
            CHECK(info.storage_buffers_descriptors[0].cbuf_offset == 0x168);
            CHECK(info.storage_buffers_descriptors[1].cbuf_offset == 0x178);
            CHECK(info.nvn_buffer_used.none());
        } else {
            CHECK(info.nvn_buffer_used.test(3));
            CHECK(info.nvn_buffer_used.test(15));
        }
        REQUIRE(std::ranges::count_if(fixture.block.Instructions(), [](const IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::GlobalAtomicIAdd32;
        }) == 1);
        const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
        const auto atomics = Instructions(code, 234);
        REQUIRE(atomics.size() == (native ? 3U : 2U));
        std::map<u32, u32> constants;
        std::map<u32, u32> pointer_bases;
        std::map<u32, u32> bindings;
        for (const auto& op : Instructions(code, 43)) {
            if (op.size() == 3) constants[op[1]] = op[2];
        }
        for (const auto& op : Instructions(code, 65)) {
            if (op.size() >= 3) pointer_bases[op[1]] = op[2];
        }
        for (const auto& op : Instructions(code, 71)) {
            if (op.size() == 3 && op[1] == 33) bindings[op[0]] = op[2];
        }
        const auto returns = Instructions(code, 254);
        std::set<u32> atomic_bindings;
        for (const auto& op : atomics) {
            REQUIRE(op.size() == 6);
            CHECK(constants.at(op[3]) == 1);
            CHECK(constants.at(op[4]) == 0);
            CHECK(std::ranges::any_of(returns, [&](const auto& returned) {
                return returned.size() == 1 && returned[0] == op[1];
            }));
            atomic_bindings.insert(bindings.at(pointer_bases.at(op[2])));
        }
        CHECK(atomic_bindings == (native ? std::set<u32>{2, 3, 4} : std::set<u32>{3, 4}));
    }
}

TEST_CASE("Global loads do not emit unused atomic helpers", "[shader][spirv][global_atomic]") {
    GlobalAtomicProgram fixture{true, 4, false};
    const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
    CHECK(Instructions(code, 234).empty());
}

TEST_CASE("Global unsigned atomic max routes buffers and returns the old value",
          "[shader][spirv][global_atomic][oot]") {
    for (const bool native : {false, true}) {
        for (const bool aliases : {false, true}) {
            CAPTURE(native, aliases);
            GlobalAtomicProgram fixture{native};
            fixture.profile.support_descriptor_aliasing = aliases;
            for (auto& inst : fixture.block.Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::GlobalAtomicIAdd32) {
                    inst.ReplaceOpcode(IR::Opcode::GlobalAtomicUMax32);
                }
            }
            const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
            const auto atomics = Instructions(code, 239);
            REQUIRE(atomics.size() == (native ? 3U : 2U));
            CHECK(Instructions(code, 234).empty());
            const auto returns = Instructions(code, 254);
            for (const auto& atomic : atomics) {
                REQUIRE(atomic.size() == 6);
                CHECK(std::ranges::any_of(returns, [&](const auto& returned) {
                    return returned.size() == 1 && returned[0] == atomic[1];
                }));
            }
        }
    }
}

TEST_CASE("Global atomic routing works without descriptor aliases",
          "[shader][spirv][global_atomic]") {
    for (const bool native : {false, true}) {
        GlobalAtomicProgram fixture{native};
        fixture.profile.support_descriptor_aliasing = false;
        const auto code = Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
        const auto atomics = Instructions(code, 234);
        REQUIRE(atomics.size() == (native ? 3U : 2U));
        const auto returns = Instructions(code, 254);
        for (const auto& op : atomics) {
            CHECK(std::ranges::any_of(returns, [&](const auto& returned) {
                return returned.size() == 1 && returned[0] == op[1];
            }));
        }
    }
}

TEST_CASE("Global atomic add rejects unsupported 64-bit addressing",
          "[shader][spirv][global_atomic]") {
    TestEnvironment env;
    Program fixture;
    IR::IREmitter ir{fixture.block};
    static_cast<void>(ir.GlobalAtomicIAdd(ir.Imm64(u64{0}), ir.Imm32(1)));
    Optimization::CollectShaderInfoPass(env, fixture.program);
    try {
        static_cast<void>(Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program));
        FAIL("Global atomic add silently ignored missing address support");
    } catch (const Shader::NotImplementedException& exception) {
        const std::string message{exception.what()};
        CHECK(message.find("GlobalAtomicIAdd32") != std::string::npos);
        CHECK(message.find("64-bit integers") != std::string::npos);
    }
}

TEST_CASE("Global atomic add without a routable buffer remains an explicit failure",
          "[shader][spirv][global_atomic]") {
    for (const u32 scenario : {0U, 1U, 2U, 3U}) {
        CAPTURE(scenario);
        TestEnvironment env;
        Program fixture;
        fixture.profile.support_int64 = true;
        fixture.profile.support_descriptor_aliasing = true;
        IR::IREmitter ir{fixture.block};
        static_cast<void>(ir.GlobalAtomicIAdd(ir.Imm64(u64{0x1200001000}), ir.Imm32(1)));
        Optimization::CollectShaderInfoPass(env, fixture.program);
        auto& info = fixture.program.info;
        info.native_global_bounds = scenario == 1;
        if (scenario >= 2) {
            info.storage_buffers_descriptors = {{scenario == 2 ? 1U : 0U, 0x340, 1, true}};
            info.nvn_buffer_used.set(scenario == 2 ? 3 : 15);
            info.constant_buffer_descriptors = {{0, 1}, {1, 1}};
        }
        try {
            static_cast<void>(Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program));
            FAIL("Unrouteable global atomic add silently compiled as a no-op");
        } catch (const Shader::NotImplementedException& exception) {
            const std::string message{exception.what()};
            CHECK(message.find("GlobalAtomicIAdd32") != std::string::npos);
            CHECK(message.find("no routable storage buffer") != std::string::npos);
        }
    }
}
