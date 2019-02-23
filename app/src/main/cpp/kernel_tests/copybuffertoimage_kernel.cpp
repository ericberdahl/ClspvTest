//
// Created by Eric Berdahl on 10/31/17.
//

#include "copybuffertoimage_kernel.hpp"

namespace  {
    template <typename ImagePixelType>
    test_utils::KernelTest::invocation_tests compose_test_series()
    {
        const auto tests = {
                copybuffertoimage_kernel::getTestVariant<gpu_types::uchar, ImagePixelType>(),
                copybuffertoimage_kernel::getTestVariant<gpu_types::uchar4, ImagePixelType>(),
                copybuffertoimage_kernel::getTestVariant<gpu_types::half, ImagePixelType>(),
                copybuffertoimage_kernel::getTestVariant<gpu_types::half4, ImagePixelType>(),
                copybuffertoimage_kernel::getTestVariant<float, ImagePixelType>(),
                copybuffertoimage_kernel::getTestVariant<gpu_types::float2, ImagePixelType>(),
                copybuffertoimage_kernel::getTestVariant<gpu_types::float4, ImagePixelType>(),
        };

        return test_utils::KernelTest::invocation_tests(std::begin(tests), std::end(tests));
    }
}

namespace copybuffertoimage_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&     kernel,
           vulkan_utils::buffer&    src_buffer,
           vulkan_utils::image&     dst_image,
           int                      src_offset,
           int                      src_pitch,
           cl_channel_order         src_channel_order,
           cl_channel_type          src_channel_type,
           bool                     swap_components,
           bool                     premultiply,
           int                      width,
           int                      height)
    {
        struct scalar_args {
            int inSrcOffset;        // offset 0
            int inSrcPitch;         // offset 4
            int inSrcChannelOrder;  // offset 8 -- cl_channel_order
            int inSrcChannelType;   // offset 12 -- cl_channel_type
            int inSwapComponents;   // offset 16 -- bool
            int inPremultiply;      // offset 20 -- bool
            int inWidth;            // offset 24
            int inHeight;           // offset 28
        };
        static_assert(0 == offsetof(scalar_args, inSrcOffset), "inSrcOffset offset incorrect");
        static_assert(4 == offsetof(scalar_args, inSrcPitch), "inSrcPitch offset incorrect");
        static_assert(8 == offsetof(scalar_args, inSrcChannelOrder),
                      "inSrcChannelOrder offset incorrect");
        static_assert(12 == offsetof(scalar_args, inSrcChannelType),
                      "inSrcChannelType offset incorrect");
        static_assert(16 == offsetof(scalar_args, inSwapComponents),
                      "inSwapComponents offset incorrect");
        static_assert(20 == offsetof(scalar_args, inPremultiply), "inPremultiply offset incorrect");
        static_assert(24 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(28 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        vulkan_utils::buffer scalarBuffer = vulkan_utils::createUniformBuffer(kernel.getDevice().getDevice(),
                                                                              kernel.getDevice().getMemoryProperties(),
                                                                              sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inSrcOffset = src_offset;
        scalars->inSrcPitch = src_pitch;
        scalars->inSrcChannelOrder = src_channel_order;
        scalars->inSrcChannelType = src_channel_type;
        scalars->inSwapComponents = (swap_components ? 1 : 0);
        scalars->inPremultiply = (premultiply ? 1 : 0);
        scalars->inWidth = width;
        scalars->inHeight = height;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(src_buffer);
        invocation.addWriteOnlyImageArgument(dst_image);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
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

        test_utils::KernelTest::invocation_tests result;
        for (auto& s : series) {
            test_utils::KernelTest::invocation_tests nextSeries = s();
            result.insert(result.end(), nextSeries.begin(), nextSeries.end());
        }

        return result;
    }

}
