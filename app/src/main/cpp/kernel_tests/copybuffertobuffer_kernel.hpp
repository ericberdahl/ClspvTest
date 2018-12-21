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

    struct TestBase
    {
        TestBase(const clspv_utils::device& device, const std::vector<std::string>& args, std::size_t sizeofPixelComponent, unsigned int numComponents);

        ~TestBase();

        void run(clspv_utils::kernel& kernel, test_utils::InvocationResult& invocationResult);

        vk::Extent3D                    mBufferExtent;
        vulkan_utils::storage_buffer    mSrcBuffer;
        vulkan_utils::storage_buffer    mDstBuffer;
        bool                            mIs32Bit;
    };

    template <typename PixelType>
    struct Test : private TestBase
    {
        Test(const clspv_utils::device& device, const std::vector<std::string>& args) :
            TestBase(device, args, sizeof(typename PixelType::component_type), PixelType::num_components)
        {
            static_assert(std::is_floating_point<typename PixelType::component_type>::value, "copybuffertoboffer_kernel requires floating point pixels");
            static_assert(4 == PixelType::num_components, "copybuffertoboffer_kernel requires 4-vector pixels");
            static_assert(2 == sizeof(typename PixelType::component_type) || 4 == sizeof(typename PixelType::component_type), "copybuffertoboffer_kernel requires half4 or float4 pixels");

            const std::size_t buffer_length =
                    mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

            // initialize source memory with random data
            auto srcBufferMap = mSrcBuffer.map<PixelType>();
            test_utils::fill_random_pixels<PixelType>(srcBufferMap.get(),
                                                      srcBufferMap.get() + buffer_length);
        }

        void prepare()
        {
            const std::size_t buffer_length =
                    mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

            // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
            auto srcBufferMap = mSrcBuffer.map<PixelType>();
            auto dstBufferMap = mDstBuffer.map<PixelType>();
            test_utils::copy_pixel_buffer<PixelType, PixelType>(srcBufferMap.get(),
                                                                srcBufferMap.get() +
                                                                buffer_length,
                                                                dstBufferMap.get());
            test_utils::invert_pixel_buffer<PixelType>(dstBufferMap.get(),
                                                       dstBufferMap.get() + buffer_length);
        }

        using TestBase::run;

        void checkResults(test_utils::InvocationResult& invocationResult, bool verbose)
        {
            auto srcBufferMap = mSrcBuffer.map<PixelType>();
            auto dstBufferMap = mDstBuffer.map<PixelType>();
            test_utils::check_results(srcBufferMap.get(),
                                      dstBufferMap.get(),
                                      mBufferExtent,
                                      mBufferExtent.width,
                                      verbose,
                                      invocationResult);
        }
    };

    template <typename PixelType>
    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose)
    {
        test_utils::InvocationResult invocationResult;

        Test<PixelType> t(kernel.getDevice(), args);

        t.prepare();
        t.run(kernel, invocationResult);
        t.checkResults(invocationResult, verbose);

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
