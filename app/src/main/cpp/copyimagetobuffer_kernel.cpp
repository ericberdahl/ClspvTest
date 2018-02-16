//
// Created by Eric Berdahl on 10/31/17.
//

#include "copyimagetobuffer_kernel.hpp"

namespace copyimagetobuffer_kernel {

    clspv_utils::kernel_invocation::execution_time_t
    invoke(const clspv_utils::kernel_module&   module,
           const clspv_utils::kernel&          kernel,
           const sample_info&                  info,
           vk::ArrayProxy<const vk::Sampler>   samplers,
           vk::ImageView                       src_image,
           vk::Buffer                          dst_buffer,
           int                                 dst_offset,
           int                                 dst_pitch,
           cl_channel_order                    dst_channel_order,
           cl_channel_type                     dst_channel_type,
           bool                                swap_components,
           int                                 width,
           int                                 height)
    {
        struct scalar_args {
            int inDestOffset;       // offset 0
            int inDestPitch;        // offset 4
            int inDestChannelOrder; // offset 8 -- cl_channel_order
            int inDestChannelType;  // offset 12 -- cl_channel_type
            int inSwapComponents;   // offset 16 -- bool
            int inWidth;            // offset 20
            int inHeight;           // offset 24
        };
        static_assert(0 == offsetof(scalar_args, inDestOffset), "inDestOffset offset incorrect");
        static_assert(4 == offsetof(scalar_args, inDestPitch), "inDestPitch offset incorrect");
        static_assert(8 == offsetof(scalar_args, inDestChannelOrder),
                      "inDestChannelOrder offset incorrect");
        static_assert(12 == offsetof(scalar_args, inDestChannelType),
                      "inDestChannelType offset incorrect");
        static_assert(16 == offsetof(scalar_args, inSwapComponents),
                      "inSwapComponents offset incorrect");
        static_assert(20 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(24 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        const scalar_args scalars = {
                dst_offset,
                width,
                dst_channel_order,
                dst_channel_type,
                (swap_components ? 1 : 0),
                width,
                height
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
                (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                                  info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addReadOnlyImageArgument(src_image);
        invocation.addBufferArgument(dst_buffer);
        invocation.addPodArgument(scalars);

        return invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    void test_matrix(const clspv_utils::kernel_module&     module,
                     const clspv_utils::kernel&            kernel,
                     const sample_info&                    info,
                     vk::ArrayProxy<const vk::Sampler>     samplers,
                     test_utils::InvocationResultSet&      resultSet)
    {
        const test_utils::test_kernel_fn tests[] = {
                test_series<gpu_types::float4>,
                test_series<gpu_types::half4>,
                test_series<gpu_types::uchar4>,
                test_series<gpu_types::float2>,
                test_series<gpu_types::half2>,
                test_series<gpu_types::uchar2>,
                test_series<float>,
                test_series<gpu_types::half>,
                test_series<gpu_types::uchar>,
        };

        test_utils::test_kernel_invocations(module, kernel,
                                            std::begin(tests), std::end(tests),
                                            info,
                                            samplers,
                                            resultSet);
    }

}
