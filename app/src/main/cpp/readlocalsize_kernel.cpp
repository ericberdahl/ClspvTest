//
// Created by Eric Berdahl on 10/31/17.
//

#include "readlocalsize_kernel.hpp"

namespace readlocalsize_kernel {

    clspv_utils::execution_time_t
    invoke(const clspv_utils::kernel_module&    module,
           const clspv_utils::kernel&           kernel,
           const sample_info&                   info,
           vk::ArrayProxy<const vk::Sampler>    samplers,
           vk::Buffer                           outLocalSizes,
           int                                  inWidth,
           int                                  inHeight,
           int                                  inPitch,
           idtype_t                             inIdType) {
        struct scalar_args {
            int width;  // offset 0
            int height; // offset 4
            int pitch;  // offset 8
            int idtype; // offset 12
        };
        static_assert(0 == offsetof(scalar_args, width), "width offset incorrect");
        static_assert(4 == offsetof(scalar_args, height), "height offset incorrect");
        static_assert(8 == offsetof(scalar_args, pitch), "pitch offset incorrect");
        static_assert(12 == offsetof(scalar_args, idtype), "idtype offset incorrect");

        const scalar_args scalars = {
                inWidth,
                inHeight,
                inPitch,
                inIdType
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (scalars.width + workgroup_sizes.x - 1) / workgroup_sizes.x,
                (scalars.height + workgroup_sizes.y - 1) / workgroup_sizes.y);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                              info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addStorageBufferArgument(outLocalSizes);
        invocation.addPodArgument(scalars);

        return invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    void test(const clspv_utils::kernel_module&  module,
              const clspv_utils::kernel&         kernel,
              const sample_info&                 info,
              vk::ArrayProxy<const vk::Sampler>  samplers,
              const std::vector<std::string>&    args,
              bool                               verbose,
              test_utils::InvocationResultSet&   resultSet)
    {
        test_utils::InvocationResult    invocationResult;

        invocationResult.mVariation = "global_id_x";

        int buffer_height   = 64;
        int buffer_width    = 64;

        // allocate data buffer
        const std::size_t buffer_size = buffer_width * buffer_height * sizeof(std::int32_t);
        vulkan_utils::storage_buffer dst_buffer(info, buffer_size);

        std::vector<std::int32_t> expectedResults(buffer_height * buffer_width);
        auto rowIter = expectedResults.begin();
        for (int i = 0; i < buffer_height; ++i) {
            auto nextRow = std::next(rowIter, buffer_width);
            std::iota(rowIter, nextRow, 0);
            rowIter = nextRow;
        }

        invocationResult.mExecutionTime = invoke(module,
                                                 kernel,
                                                 info,
                                                 samplers,
                                                 *dst_buffer.buf,
                                                 buffer_width,
                                                 buffer_height,
                                                 buffer_width,
                                                 idtype_globalid_x);

        test_utils::check_results<std::int32_t,std::int32_t>(expectedResults.data(),
                                                             dst_buffer.mem,
                                                             buffer_width,
                                                             buffer_height,
                                                             buffer_width,
                                                             verbose,
                                                             invocationResult);

        resultSet.push_back(std::move(invocationResult));
    };
}
