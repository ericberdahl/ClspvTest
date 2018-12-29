//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_FILLARRAYSTRUCT_KERNEL_HPP
#define CLSPVTEST_FILLARRAYSTRUCT_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace fillarraystruct_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    destination_buffer,
           unsigned int                     num_elements);

    struct Test
    {
        Test(const clspv_utils::device& device, const std::vector<std::string>& args);

        void prepare();

        clspv_utils::execution_time_t run(clspv_utils::kernel& kernel);

        test_utils::Evaluation checkResults(bool verbose);

        static const unsigned int kWrapperArraySize = 18;

        typedef struct {
            float arr[kWrapperArraySize];
        } FloatArrayWrapper;

        int                             mBufferWidth;
        vulkan_utils::storage_buffer    mDstBuffer;
        std::vector<FloatArrayWrapper>  mExpectedResults;
    };

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose);

    test_utils::KernelTest::invocation_tests getAllTestVariants();
}

#endif //CLSPVTEST_FILLARRAYSTRUCT_KERNEL_HPP
