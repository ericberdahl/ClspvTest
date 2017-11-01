//
// Created by Eric Berdahl on 10/31/17.
//

#include "fill_kernel.hpp"

namespace fill_kernel {

    void invoke(const clspv_utils::kernel_module &module,
                const clspv_utils::kernel &kernel,
                const sample_info &info,
                const std::vector<VkSampler> &samplers,
                VkBuffer dst_buffer,
                int pitch,
                int device_format,
                int offset_x,
                int offset_y,
                int width,
                int height,
                const gpu_types::float4 &color) {
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

        const scalar_args scalars = {
                pitch,
                device_format,
                offset_x,
                offset_y,
                width,
                height,
                color
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (scalars.inWidth + workgroup_sizes.x - 1) / workgroup_sizes.x,
                (scalars.inHeight + workgroup_sizes.y - 1) / workgroup_sizes.y);

        clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool,
                                                  info.memory_properties);

        invocation.addLiteralSamplers(samplers.begin(), samplers.end());
        invocation.addBufferArgument(dst_buffer);
        invocation.addPodArgument(scalars);
        invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    test_utils::Results test_series(const clspv_utils::kernel_module&  module,
                                    const clspv_utils::kernel&         kernel,
                                    const sample_info&                 info,
                                    const std::vector<VkSampler>&      samplers,
                                    const test_utils::options&         opts) {
        const test_utils::test_kernel_fn tests[] = {
                test<gpu_types::float4>,
                test<gpu_types::half4>,
        };

        return test_utils::test_kernel_invocation(module,
                                                  kernel,
                                                  std::begin(tests), std::end(tests),
                                                  info,
                                                  samplers,
                                                  opts);
    }

}
