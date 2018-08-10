//
// Created by Eric Berdahl on 10/31/17.
//

#include "copyimagetobuffer_kernel.hpp"

namespace  {
    template <typename ImagePixelType>
    test_utils::test_kernel_series compose_test_series()
    {
        const auto tests = {
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<gpu_types::uchar, ImagePixelType>),
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<gpu_types::uchar4, ImagePixelType>),
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<gpu_types::half, ImagePixelType>),
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<gpu_types::half4, ImagePixelType>),
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<float, ImagePixelType>),
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<gpu_types::float2, ImagePixelType>),
                test_utils::test_kernel_fn(copyimagetobuffer_kernel::test<gpu_types::float4, ImagePixelType>),
        };

        return test_utils::test_kernel_series(std::begin(tests), std::end(tests));
    }
}


namespace copyimagetobuffer_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              dst_offset,
           int                              dst_pitch,
           cl_channel_order                 dst_channel_order,
           cl_channel_type                  dst_channel_type,
           bool                             swap_components,
           int                              width,
           int                              height)
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

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().mDevice,
                                                  kernel.getDevice().mMemoryProperties,
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inDestOffset = dst_offset;
        scalars->inDestPitch = width;
        scalars->inDestChannelOrder = dst_channel_order;
        scalars->inDestChannelType = dst_channel_type;
        scalars->inSwapComponents = (swap_components ? 1 : 0);
        scalars->inWidth = width;
        scalars->inHeight = height;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::kernel_invocation invocation = kernel.createInvocation();

        invocation.addReadOnlyImageArgument(src_image);
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    test_utils::test_kernel_series getAllTestVariants()
    {
        const auto series = {
                compose_test_series<gpu_types::float4>,
                compose_test_series<gpu_types::half4>,
                compose_test_series<gpu_types::uchar4>,
                compose_test_series<gpu_types::float2>,
                compose_test_series<gpu_types::half2>,
                compose_test_series<gpu_types::uchar2>,
                compose_test_series<float>,
                compose_test_series<gpu_types::half>,
                compose_test_series<gpu_types::uchar>,
        };

        test_utils::test_kernel_series result;
        for (auto& s : series) {
            test_utils::test_kernel_series nextSeries = s();
            result.insert(result.end(), nextSeries.begin(), nextSeries.end());
        }

        return result;
    }

}
