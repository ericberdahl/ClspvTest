//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_FILL_KERNEL_HPP
#define CLSPVTEST_FILL_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace fill_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              pitch,
           int                              device_format,
           int                              offset_x,
           int                              offset_y,
           int                              width,
           int                              height,
           const gpu_types::float4&         color);

    test_utils::KernelTest::invocation_tests getAllTestVariants();

    template <typename PixelType>
    struct Test : public test_utils::Test
    {
        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
            mBufferExtent(64, 64, 1),
            mFillColor(0.25f, 0.50f, 0.75f, 1.0f)
        {
            auto& device = kernel.getDevice();

            for (auto arg = args.begin(); arg != args.end(); arg = std::next(arg)) {
                if (*arg == "-w") {
                    arg = std::next(arg);
                    if (arg == args.end()) throw std::runtime_error("badly formed arguments to fill test");
                    mBufferExtent.width = std::atoi(arg->c_str());
                }
                else if (*arg == "-h") {
                    arg = std::next(arg);
                    if (arg == args.end()) throw std::runtime_error("badly formed arguments to fill test");
                    mBufferExtent.height = std::atoi(arg->c_str());
                }
            }

            // allocate image buffer
            const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
            const std::size_t buffer_size = buffer_length * sizeof(PixelType);
            mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), buffer_size);
        }

        virtual void prepare() override
        {
            const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

            const PixelType src_value = pixels::traits<PixelType>::translate((gpu_types::float4){ 0.0f, 0.0f, 0.0f, 0.0f });
            auto dstBufferMap = mDstBuffer.map<PixelType>();
            std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, src_value);
        }

        virtual std::string getParameterString() const override
        {
            std::ostringstream os;
            os << "<w:" << mBufferExtent.width << " h:" << mBufferExtent.height << " d:" << mBufferExtent.depth << ">";
            return os.str();
        }

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override
        {

            return invoke(kernel,
                          mDstBuffer, // dst_buffer
                          mBufferExtent.width,   // pitch
                          pixels::traits<PixelType>::device_pixel_format, // device_format
                          0, 0, // offset_x, offset_y
                          mBufferExtent.width, mBufferExtent.height, // width, height
                          mFillColor); // color
        }

        virtual test_utils::Evaluation evaluate(bool verbose) override
        {
            auto dstBufferMap = mDstBuffer.map<PixelType>();
            return test_utils::check_results(dstBufferMap.get(),
                                             mBufferExtent,
                                             mBufferExtent.width,
                                             mFillColor,
                                             verbose);
        }

        vk::Extent3D                    mBufferExtent;
        vulkan_utils::storage_buffer    mDstBuffer;
        gpu_types::float4               mFillColor;
    };

    template <typename PixelType>
    test_utils::InvocationTest getTestVariant()
    {
        test_utils::InvocationTest result;

        std::ostringstream os;
        os << "<dst:" << pixels::traits<PixelType>::type_name << ">";
        result.mVariation = os.str();

        result.mTestFn = test_utils::run_test<Test<PixelType>>;

        return result;
    }


}

#endif //CLSPVTEST_FILL_KERNEL_HPP
