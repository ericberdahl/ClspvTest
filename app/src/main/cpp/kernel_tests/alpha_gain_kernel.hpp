//
// Created by Pervez Alam on 03/01/20.
//

#ifndef CLSPVTEST_ALPHA_GAIN_KERNEL_HPP
#define CLSPVTEST_ALPHA_GAIN_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"
#include "../../third_party/vulkan/vulkan/vulkan.hpp"
#include "clspv_utils/device.hpp"
#include "pixels.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace alpha_gain_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::image&             dst_image,
           int                              pitch,
           int                              device_format,
           int                              width,
           int                              height,
           const float                      alpha_gain_factor);

    test_utils::KernelTest::invocation_tests getAllTestVariants();

    template <typename PixelType>
    struct Test : public test_utils::Test
    {
        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
            mExtent(64, 64, 1),
            mAlphaGainFactor(0.314f),
            mDevice(kernel.getDevice())
        {
            for (auto arg = args.begin(); arg != args.end(); arg = std::next(arg)) {
                if (*arg == "-w") {
                    arg = std::next(arg);
                    if (arg == args.end()) throw std::runtime_error("badly formed arguments to fill test");
                    mExtent.width = std::atoi(arg->c_str());
                }
                else if (*arg == "-h") {
                    arg = std::next(arg);
                    if (arg == args.end()) throw std::runtime_error("badly formed arguments to fill test");
                    mExtent.height = std::atoi(arg->c_str());
                }
            }
        }

        virtual void prepare() override
        {
            // allocate image buffer
            const std::size_t buffer_length = mExtent.width * mExtent.height * mExtent.depth;
            const std::size_t buffer_size = buffer_length * sizeof(PixelType);

            // allocate buffers and images
            mSrcImage = vulkan_utils::image(mDevice.getDevice(),
                                            mDevice.getMemoryProperties(),
                                            vk::Extent3D(mExtent.width, mExtent.height, 1),
                                            vk::Format(pixels::traits<PixelType>::vk_pixel_type),
                                            vulkan_utils::image::kUsage_ReadOnly);

            mSrcImageStaging = vulkan_utils::createStagingBuffer(mDevice.getDevice(),
                                                                 mDevice.getMemoryProperties(),
                                                                 mSrcImage,
                                                                 true,
                                                                 false);

            mDstImage = vulkan_utils::image(mDevice.getDevice(),
                                            mDevice.getMemoryProperties(),
                                            vk::Extent3D(mExtent.width, mExtent.height, 1),
                                            vk::Format(pixels::traits<PixelType>::vk_pixel_type),
                                            vulkan_utils::image::kUsage_ReadWrite);

            mDstImageStaging = vulkan_utils::createStagingBuffer(mDevice.getDevice(),
                                                                 mDevice.getMemoryProperties(),
                                                                 mDstImage,
                                                                 true,
                                                                 false);

            // initialize source memory with random data
            auto srcImageMap = mSrcImageStaging.map<PixelType>();

            gpu_types::float4 *image_buffer_data = new gpu_types::float4[buffer_length];
            for ( int i = 0; i < buffer_length; ++i)
            {
                gpu_types::float4 pixel = {(float)i / (float) buffer_length / 16.0f, (float)i / (float) buffer_length / 8.0f, (float)i / (float) buffer_length / 4.0f, 0.9 };
                image_buffer_data[i] = pixel;
            }

            std::copy(image_buffer_data, image_buffer_data + buffer_length, srcImageMap.get());
            srcImageMap.reset();

            mExpectedDstBuffer.resize(buffer_length);
            //TODO: Add expected result
//            for (int row = 0; row < mExtent.height; ++row)
//            {
//                for (int col = 0; col < mExtent.width; ++col)
//                {
//                    gpu_types::float2 normalizedCoordinate(((float)col + 0.5f) / ((float)mExtent.width),
//                                                           ((float)row + 0.5f) / ((float)mExtent.height));
//
//                    gpu_types::float2 sampledCoordinate(clampf(normalizedCoordinate.x*image_width - 0.5f, 0.0f, image_width - 1)/(image_width - 1),
//                                                        clampf(normalizedCoordinate.y*image_height - 0.5f, 0.0f, image_height - 1)/(image_height - 1));
//
//                    auto index = (row * mExtent.width) + col;
//                    mExpectedDstBuffer[index] = BufferPixelType(sampledCoordinate.x,
//                                                                sampledCoordinate.y,
//                                                                0.0f,
//                                                                0.0f);
//                }
//            }
            delete [] image_buffer_data;
        }

        virtual std::string getParameterString() const override
        {
            std::ostringstream os;
            os << "<w:" << mExtent.width << " h:" << mExtent.height << " d:" << mExtent.depth << ">";
            return os.str();
        }

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override
        {
            return invoke(kernel,
                          mSrcImage,
                          mDstImage,
                          mExtent.width,//TODO: Do we need to maintain a pitch separately?
                          pixels::traits<PixelType>::device_pixel_format,
                          mExtent.width,
                          mExtent.height,
                          mAlphaGainFactor);
        }

        virtual test_utils::Evaluation evaluate(bool verbose) override
        {
            auto dstBufferMap = mDstImageStaging.map<PixelType>();
            return test_utils::check_results(dstBufferMap.get(),
                                             mExtent,
                                             mExtent.width,
                                             mAlphaGainFactor,
                                             verbose);
        }

        clspv_utils::device     mDevice;
        vk::Extent3D            mExtent;
        vulkan_utils::image     mDstImage;
        vulkan_utils::buffer    mDstImageStaging;
        vulkan_utils::image     mSrcImage;
        vulkan_utils::buffer    mSrcImageStaging;
        float                   mAlphaGainFactor;
        std::vector<PixelType>  mExpectedDstBuffer;
    };

    template <typename PixelType>
    test_utils::InvocationTest getTestVariant()
    {
        std::ostringstream os;
        os << "<dst:" << pixels::traits<PixelType>::type_name << ">";

        return test_utils::make_invocation_test< Test<PixelType> >(os.str());
    }


}

#endif //CLSPVTEST_ALPHA_GAIN_KERNEL_HPP
