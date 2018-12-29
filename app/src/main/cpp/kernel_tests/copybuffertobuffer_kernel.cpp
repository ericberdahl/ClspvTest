//
// Created by Eric Berdahl on 10/31/17.
//

#include "copybuffertobuffer_kernel.hpp"

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
           std::int32_t                     height)
    {
        struct scalar_args {
            std::int32_t inSrcPitch;         // offset 0
            std::int32_t inSrcOffset;        // offset 4
            std::int32_t inDstPitch;         // offset 8
            std::int32_t inDstOffset;        // offset 12
            std::int32_t inIs32Bit;          // offset 16
            std::int32_t inWidth;            // offset 20
            std::int32_t inHeight;           // offset 24
        };
        static_assert(0 == offsetof(scalar_args, inSrcPitch), "inSrcPitch offset incorrect");
        static_assert(4 == offsetof(scalar_args, inSrcOffset), "inSrcOffset offset incorrect");
        static_assert(8 == offsetof(scalar_args, inDstPitch), "inDstPitch offset incorrect");
        static_assert(12 == offsetof(scalar_args, inDstOffset), "inDstOffset offset incorrect");
        static_assert(16 == offsetof(scalar_args, inIs32Bit), "inIs32Bit offset incorrect");
        static_assert(20 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(24 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().getDevice(),
                                                  kernel.getDevice().getMemoryProperties(),
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inSrcPitch = src_pitch;
        scalars->inSrcOffset = src_offset;
        scalars->inDstPitch = dst_pitch;
        scalars->inDstOffset = dst_offset;
        scalars->inIs32Bit = is32Bit;
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
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        const auto test_variants = {
                getTestVariant<gpu_types::float4>(),
                getTestVariant<gpu_types::half4>(),
        };

        return test_utils::KernelTest::invocation_tests(test_variants);
    }

    TestBase::TestBase(const clspv_utils::device& device, const std::vector<std::string>& args, std::size_t sizeofPixelComponent, unsigned int numComponents) :
            mBufferExtent(64, 64, 1)
    {
        for (auto arg = args.begin(); arg != args.end(); arg = std::next(arg)) {
            if (*arg == "-w") {
                arg = std::next(arg);
                if (arg == args.end()) throw std::runtime_error("badly formed arguments to copybuffertoboffer test");
                mBufferExtent.width = std::atoi(arg->c_str());
            }
            else if (*arg == "-h") {
                arg = std::next(arg);
                if (arg == args.end()) throw std::runtime_error("badly formed arguments to copybuffertoboffer test");
                mBufferExtent.height = std::atoi(arg->c_str());
            }
        }

        const std::size_t buffer_length =
                mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeofPixelComponent * numComponents;
        mIs32Bit = (sizeofPixelComponent == 4);

        // allocate buffers and images
        mSrcBuffer = vulkan_utils::storage_buffer(device.getDevice(),
                                                  device.getMemoryProperties(),
                                                  buffer_size);
        mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(),
                                                  device.getMemoryProperties(),
                                                  buffer_size);


    }

    TestBase::~TestBase()
    {

    }

    clspv_utils::execution_time_t TestBase::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mSrcBuffer,
                      mDstBuffer,
                      mBufferExtent.width,  // src_pitch
                      0,                    // src_offset
                      mBufferExtent.width,  // dst_pitch
                      0,                    // dst_offset
                      mIs32Bit,             // is32Bit
                      mBufferExtent.width,  // width
                      mBufferExtent.height);// height
    }


}
