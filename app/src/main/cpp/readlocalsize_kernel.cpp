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

        invocation.addLiteralSamplers(samplers);
        invocation.addBufferArgument(*outArgs.buf);

        invocation.run(info.graphics_queue, kernel, num_workgroups);

        vulkan_utils::memory_map argMap(outArgs);
        auto outScalars = static_cast<const scalar_args *>(argMap.map());

        const auto result = std::make_tuple(outScalars->outWorkgroupX,
                                            outScalars->outWorkgroupY,
                                            outScalars->outWorkgroupZ);

        return result;
    }

    void test(const clspv_utils::kernel_module&  module,
              const clspv_utils::kernel&         kernel,
              const sample_info&                 info,
              vk::ArrayProxy<const vk::Sampler>  samplers,
              test_utils::InvocationResultSet&   resultSet)
    {
        test_utils::InvocationResult    invocationResult;

        const clspv_utils::WorkgroupDimensions expected = kernel.getWorkgroupSize();

        const auto observed = invoke(module, kernel, info, samplers);

        const bool success = (expected.x == std::get<0>(observed) &&
                              expected.y == std::get<1>(observed) &&
                              1 == std::get<2>(observed));

        if (success) {
            ++invocationResult.mNumCorrectPixels;
        } else {
            std::ostringstream os;
            os << (success ? "CORRECT" : "INCORRECT")
               << ": workgroup_size expected{x=" << expected.x << ", y=" << expected.y << ", z=1}"
               << " observed{x=" << std::get<0>(observed) << ", y=" << std::get<1>(observed) <<", z=" << std::get<2>(observed) << "}";
            invocationResult.mPixelErrors.push_back(os.str());
        }

        resultSet.push_back(invocationResult);
    };
}
