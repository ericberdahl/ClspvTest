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
    invoke(clspv_utils::kernel&     kernel,
           vulkan_utils::buffer&    src_buffer,
           vulkan_utils::buffer&    dst_buffer,
           std::int32_t             src_pitch,
           std::int32_t             src_offset,
           std::int32_t             dst_pitch,
           std::int32_t             dst_offset,
           bool                     is32Bit,
           std::int32_t             width,
           std::int32_t             height);

    test_utils::KernelTest::invocation_tests getAllTestVariants();

    struct TestBase : public test_utils::Test
    {
        TestBase(clspv_utils::kernel& kernel, const std::vector<std::string>& args, std::size_t sizeofPixelComponent, unsigned int numComponents);

        ~TestBase();

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override;

        vk::Extent3D            mBufferExtent;
        vulkan_utils::buffer    mSrcBuffer;
        vulkan_utils::buffer    mDstBuffer;
        bool                    mIs32Bit;
    };

    template <typename PixelType>
    struct Test : public TestBase
    {
        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
            TestBase(kernel, args, sizeof(typename PixelType::component_type), PixelType::num_components)
        {
            auto& device = kernel.getDevice();

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

        virtual void prepare() override
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

        virtual test_utils::Evaluation evaluate(bool verbose) override
        {
            auto srcBufferMap = mSrcBuffer.map<PixelType>();
            auto dstBufferMap = mDstBuffer.map<PixelType>();
            return test_utils::check_results(srcBufferMap.get(),
                                             dstBufferMap.get(),
                                             mBufferExtent,
                                             mBufferExtent.width,
                                             verbose);
        }
    };

    template <typename PixelType>
    test_utils::InvocationTest getTestVariant()
    {
        std::ostringstream os;
        os << "<pixelType:" << pixels::traits<PixelType>::type_name << ">";

        return test_utils::make_invocation_test< Test<PixelType> >(os.str());
    }
}

#endif //CLSPVTEST_COPYBUFFERTOBUFFER_KERNEL_HPP
