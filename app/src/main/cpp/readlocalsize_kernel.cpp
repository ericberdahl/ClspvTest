//
// Created by Eric Berdahl on 10/31/17.
//

#include "readlocalsize_kernel.hpp"

namespace readlocalsize_kernel {

    clspv_utils::execution_time_t
    invoke(const clspv_utils::kernel_module&   module,
           const clspv_utils::kernel&          kernel,
           const sample_info&                  info,
           vk::ArrayProxy<const vk::Sampler>   samplers,
           std::tuple<int, int, int>&          outLocalSizes) {
        struct scalar_args {
            int outWorkgroupX;  // offset 0
            int outWorkgroupY;  // offset 4
            int outWorkgroupZ;  // offset 8
        };
        static_assert(0 == offsetof(scalar_args, outWorkgroupX), "outWorkgroupX offset incorrect");
        static_assert(4 == offsetof(scalar_args, outWorkgroupY), "outWorkgroupY offset incorrect");
        static_assert(8 == offsetof(scalar_args, outWorkgroupZ), "outWorkgroupZ offset incorrect");

        vulkan_utils::storage_buffer outArgs(info, sizeof(scalar_args));

        // The localsize kernel needs only a single workgroup with a single workitem
        const clspv_utils::WorkgroupDimensions num_workgroups(1, 1);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                              info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addStorageBufferArgument(*outArgs.buf);

        auto result = invocation.run(info.graphics_queue, kernel, num_workgroups);

        scalar_args outScalars;
        vulkan_utils::copyFromDeviceMemory(&outScalars, outArgs.mem, sizeof(outScalars));
        outLocalSizes = std::make_tuple(outScalars.outWorkgroupX,
                                        outScalars.outWorkgroupY,
                                        outScalars.outWorkgroupZ);

        return result;
    }

    void test(const clspv_utils::kernel_module&  module,
              const clspv_utils::kernel&         kernel,
              const sample_info&                 info,
              vk::ArrayProxy<const vk::Sampler>  samplers,
              const std::vector<std::string>&    args,
              test_utils::InvocationResultSet&   resultSet)
    {
        test_utils::InvocationResult    invocationResult;

        const clspv_utils::WorkgroupDimensions expected = kernel.getWorkgroupSize();

        std::tuple<int, int, int> observed;
        invocationResult.mExecutionTime = invoke(module, kernel, info, samplers, observed);

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

        resultSet.push_back(std::move(invocationResult));
    };
}
