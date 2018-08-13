//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_STRANGESHUFFLE_KERNEL_HPP
#define CLSPVTEST_STRANGESHUFFLE_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace strangeshuffle_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    index_buffer,
           vulkan_utils::storage_buffer&    source_buffer,
           vulkan_utils::storage_buffer&    destination_buffer,
           std::size_t                      num_elements);

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose);

    test_utils::KernelTest::invocation_tests getAllTestVariants();
}

#endif //CLSPVTEST_STRANGESHUFFLE_KERNEL_HPP
