//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_COPYBUFFERTOBUFFER_KERNEL_HPP
#define CLSPVTEST_COPYBUFFERTOBUFFER_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace copybuffertobuffer_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    src_buffer,
           vulkan_utils::storage_buffer&    dst_buffer,
           std::int32_t                     src_pitch,
           std::int32_t                     src_offset,
           std::int32_t                     dst_pitch,
           std::int32_t                     dst_offset,
           bool                             is32Bit,
           std::int32_t                     width,
           std::int32_t                     height);

    test_utils::KernelTest::invocation_tests getAllTestVariants();

    template <typename PixelType>
    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose)
    {
        static_assert(std::is_floating_point<typename PixelType::component_type>::value, "copybuffertoboffer_kernel requires floating point pixels");
        static_assert(4 == PixelType::num_components, "copybuffertoboffer_kernel requires 4-vector pixels");
        static_assert(2 == sizeof(typename PixelType::component_type) || 4 == sizeof(typename PixelType::component_type), "copybuffertoboffer_kernel requires half4 or float4 pixels");

        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        vk::Extent3D bufferExtent(64, 64, 1);

        for (auto arg = args.begin(); arg != args.end(); arg = std::next(arg)) {
            if (*arg == "-w") {
                arg = std::next(arg);
                if (arg == args.end()) throw std::runtime_error("badly formed arguments to copybuffertoboffer test");
                bufferExtent.width = std::atoi(arg->c_str());
            }
            else if (*arg == "-h") {
                arg = std::next(arg);
                if (arg == args.end()) throw std::runtime_error("badly formed arguments to copybuffertoboffer test");
                bufferExtent.height = std::atoi(arg->c_str());
            }
        }

        const std::size_t buffer_length =
                bufferExtent.width * bufferExtent.height * bufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(PixelType);
        const bool is32Bit = (sizeof(typename PixelType::component_type) == 4);

        // allocate buffers and images
        vulkan_utils::storage_buffer srcBuffer(device.getDevice(),
                                               device.getMemoryProperties(),
                                               buffer_size);
        vulkan_utils::storage_buffer dstBuffer(device.getDevice(),
                                               device.getMemoryProperties(),
                                               buffer_size);

        // initialize source memory with random data
        auto srcBufferMap = srcBuffer.map<PixelType>();
        test_utils::fill_random_pixels<PixelType>(srcBufferMap.get(),
                                                  srcBufferMap.get() + buffer_length);

        // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
        auto dstBufferMap = dstBuffer.map<PixelType>();
        test_utils::copy_pixel_buffer<PixelType, PixelType>(srcBufferMap.get(),
                                                            srcBufferMap.get() +
                                                            buffer_length,
                                                            dstBufferMap.get());
        test_utils::invert_pixel_buffer<PixelType>(dstBufferMap.get(),
                                                   dstBufferMap.get() + buffer_length);

        dstBufferMap.reset();
        srcBufferMap.reset();

        invocationResult.mExecutionTime = invoke(kernel,
                                                 srcBuffer,
                                                 dstBuffer,
                                                 bufferExtent.width,   // src_pitch
                                                 0,                    // src_offset
                                                 bufferExtent.width,   // dst_pitch
                                                 0,                    // dst_offset
                                                 is32Bit,              // is32Bit
                                                 bufferExtent.width,   // width
                                                 bufferExtent.height); // height

        srcBufferMap = srcBuffer.map<PixelType>();
        dstBufferMap = dstBuffer.map<PixelType>();
        test_utils::check_results(srcBufferMap.get(),
                                  dstBufferMap.get(),
                                  bufferExtent,
                                  bufferExtent.width,
                                  verbose,
                                  invocationResult);

        return invocationResult;
    }

    template <typename PixelType>
    test_utils::InvocationTest getTestVariant()
    {
        test_utils::InvocationTest result;

        std::ostringstream os;
        os << "<pixelType:" << pixels::traits<PixelType>::type_name << ">";
        result.mVariation = os.str();

        result.mTestFn = test<PixelType>;

        return result;
    }
}

#endif //CLSPVTEST_COPYBUFFERTOBUFFER_KERNEL_HPP
