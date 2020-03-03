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
           vulkan_utils::buffer&            dst_buffer,
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

            mDstBuffer = vulkan_utils::createStorageBuffer(mDevice.getDevice(),
                                                           mDevice.getMemoryProperties(),
                                                           buffer_size);

            // initialize source memory with random data
            auto srcImageMap = mSrcImageStaging.map<PixelType>();
            auto row = srcImageMap.get();
            vk::Extent3D coord;
            for (coord.height = 0; coord.height < mExtent.height; ++coord.height, row += mExtent.width)
            {
                auto p = row;
                for (coord.width = 0; coord.width < mExtent.width; ++coord.width, ++p)
                {
                    p->x = ((float)coord.width)/((float)mExtent.width);
                    p->y = ((float)coord.height)/((float)mExtent.height);
                    p->z = ((float)coord.height*coord.width)/((float)mExtent.height*mExtent.width);
                    p->w = 0.9f;
                }
            }
            srcImageMap.reset();
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
                          mDstBuffer,
                          mExtent.width * sizeof(PixelType),//TODO: Do we need to maintain a pitch separately?
                          pixels::traits<PixelType>::device_pixel_format,
                          mExtent.width,
                          mExtent.height,
                          mAlphaGainFactor);
        }

        virtual test_utils::Evaluation evaluate(bool verbose) override
        {
            auto dstBufferMap = mDstBuffer.map<PixelType>();
            test_utils::Evaluation result;

            auto row = dstBufferMap.get();
            vk::Extent3D coord;
            bool testPass = true;
            const float expectedAlpha = 0.314f * 0.9f;
            for (coord.height = 0; coord.height < mExtent.height; ++coord.height, row += mExtent.width)
            {
                auto p = row;
                for (coord.width = 0; coord.width < mExtent.width; ++coord.width, ++p)
                {
                     testPass = (p->w == expectedAlpha);
                     if (!testPass) break;
                }
                if (!testPass) break;
            }

            if (!testPass)
            {
                result.mNumErrors++;
            } else
            {
                result.mNumCorrect++;
            }

            dstBufferMap.reset();
            return result;
        }

        clspv_utils::device     mDevice;
        vk::Extent3D            mExtent;
        vulkan_utils::buffer    mDstBuffer;
        vulkan_utils::image     mSrcImage;
        vulkan_utils::buffer    mSrcImageStaging;
        float                   mAlphaGainFactor;
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
