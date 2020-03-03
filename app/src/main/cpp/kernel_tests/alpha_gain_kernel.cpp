//
// Created by Pervez Alam on 03/01/20.
//

#include "alpha_gain_kernel.hpp"

namespace alpha_gain_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::buffer&            dst_buffer,
           int                              pitch,
           int                              device_format,
           int                              width,
           int                              height,
           const float                      alpha_gain_factor) {
        struct scalar_args {
            int inPitch;                    // offset 0
            int inDeviceFormat;             // DevicePixelFormat offset 4
            int inWidth;                    // offset 8
            int inHeight;                   // offset 12
            float inAlphaGainFactor;        // offset 16
        };
        static_assert(0 == offsetof(scalar_args, inPitch), "inPitch offset incorrect");
        static_assert(4 == offsetof(scalar_args, inDeviceFormat), "inDeviceFormat offset incorrect");
        static_assert(8 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(12 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");
        static_assert(16 == offsetof(scalar_args, inAlphaGainFactor), "inAlphaGainFactor offset incorrect");

        vulkan_utils::buffer scalarBuffer = vulkan_utils::createUniformBuffer(kernel.getDevice().getDevice(),
                                                                              kernel.getDevice().getMemoryProperties(),
                                                                              sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inPitch = pitch;
        scalars->inDeviceFormat = device_format;
        scalars->inWidth = width;
        scalars->inHeight = height;
        scalars->inAlphaGainFactor = alpha_gain_factor;
        scalars.reset();

        const auto num_workgroups = vulkan_utils::computeNumberWorkgroups(kernel.getWorkgroupSize(),
                                                                          vk::Extent3D(width, height, 1));

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addReadOnlyImageArgument(src_image);
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);
        return invocation.run(num_workgroups);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        const auto test_variants = {
                getTestVariant<gpu_types::float4>()
        };

        return test_utils::KernelTest::invocation_tests(test_variants);
    }

}
