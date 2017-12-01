//
// Created by Eric Berdahl on 10/31/17.
//

#include "readlocalsize_kernel.hpp"

namespace readlocalsize_kernel {

    std::tuple<int, int, int> invoke(const clspv_utils::kernel_module&  module,
                                     const clspv_utils::kernel&         kernel,
                                     const sample_info&                 info,
                                     vk::ArrayProxy<const vk::Sampler>  samplers) {
        struct scalar_args {
            int outWorkgroupX;  // offset 0
            int outWorkgroupY;  // offset 4
            int outWorkgroupZ;  // offset 8
        };
        static_assert(0 == offsetof(scalar_args, outWorkgroupX), "outWorkgroupX offset incorrect");
        static_assert(4 == offsetof(scalar_args, outWorkgroupY), "outWorkgroupY offset incorrect");
        static_assert(8 == offsetof(scalar_args, outWorkgroupZ), "outWorkgroupZ offset incorrect");

        vulkan_utils::buffer outArgs(info, sizeof(scalar_args));

        // The localsize kernel needs only a single workgroup with a single workitem
        const clspv_utils::WorkgroupDimensions num_workgroups(1, 1);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                                  info.memory_properties);

        invocation.addBufferArgument(*outArgs.buf);

        invocation.run(info.graphics_queue, kernel, num_workgroups);

        vulkan_utils::memory_map argMap(outArgs);
        auto outScalars = static_cast<const scalar_args *>(argMap.map());

        const auto result = std::make_tuple(outScalars->outWorkgroupX,
                                            outScalars->outWorkgroupY,
                                            outScalars->outWorkgroupZ);

        return result;
    }

    test_utils::Results test(const clspv_utils::kernel_module&  module,
                             const clspv_utils::kernel&         kernel,
                             const sample_info&                 info,
                             vk::ArrayProxy<const vk::Sampler>  samplers,
                             const test_utils::options&         opts)
    {
        const clspv_utils::WorkgroupDimensions expected = kernel.getWorkgroupSize();

        const auto observed = invoke(module, kernel, info, samplers);

        const bool success = (expected.x == std::get<0>(observed) &&
                              expected.y == std::get<1>(observed) &&
                              1 == std::get<2>(observed));

        if (opts.logVerbose && ((success && opts.logCorrect) || (!success && opts.logIncorrect))) {
            const std::string label = module.getName() + "/" + kernel.getEntryPoint();
            LOGE("%s: %s workgroup_size expected{x=%d, y=%d, z=1} observed{x=%d, y=%d, z=%d}",
                 success ? "CORRECT" : "INCORRECT",
                 label.c_str(),
                 expected.x, expected.y,
                 std::get<0>(observed), std::get<1>(observed), std::get<2>(observed));
        }

        return (success ? test_utils::Results::sTestSuccess : test_utils::Results::sTestFailure);
    };
}
