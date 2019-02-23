//
// Created by Eric Berdahl on 10/31/17.
//

#include "fill_kernel.hpp"

namespace fill_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&     kernel,
           vulkan_utils::buffer&    dst_buffer,
           int                      pitch,
           int                      device_format,
           int                      offset_x,
           int                      offset_y,
           int                      width,
           int                      height,
           const gpu_types::float4& color) {
        struct scalar_args {
            int inPitch;        // offset 0
            int inDeviceFormat; // DevicePixelFormat offset 4
            int inOffsetX;      // offset 8
            int inOffsetY;      // offset 12
            int inWidth;        // offset 16
            int inHeight;       // offset 20
            gpu_types::float4 inColor;        // offset 32
        };
        static_assert(0 == offsetof(scalar_args, inPitch), "inPitch offset incorrect");
        static_assert(4 == offsetof(scalar_args, inDeviceFormat),
                      "inDeviceFormat offset incorrect");
        static_assert(8 == offsetof(scalar_args, inOffsetX), "inOffsetX offset incorrect");
        static_assert(12 == offsetof(scalar_args, inOffsetY), "inOffsetY offset incorrect");
        static_assert(16 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(20 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");
        static_assert(32 == offsetof(scalar_args, inColor), "inColor offset incorrect");

        vulkan_utils::buffer scalarBuffer = vulkan_utils::createUniformBuffer(kernel.getDevice().getDevice(),
                                                                              kernel.getDevice().getMemoryProperties(),
                                                                              sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inPitch = pitch;
        scalars->inDeviceFormat = device_format;
        scalars->inOffsetX = offset_x;
        scalars->inOffsetY = offset_y;
        scalars->inWidth = width;
        scalars->inHeight = height;
        scalars->inColor = color;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);
        return invocation.run(num_workgroups);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        const auto test_variants = {
                getTestVariant<gpu_types::float4>(),
                getTestVariant<gpu_types::half4>()
        };

        return test_utils::KernelTest::invocation_tests(test_variants);
    }

}
