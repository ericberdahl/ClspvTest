//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_TESTGREATERTHANOREQUALTO_KERNEL_HPP
#define CLSPVTEST_TESTGREATERTHANOREQUALTO_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace testgreaterthanorequalto_kernel {

    clspv_utils::execution_time_t
    invoke(const clspv_utils::kernel&       kernel,
           vulkan_utils::storage_buffer&    dst_buffer,
           vk::Extent3D                     extent);

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose);

    test_utils::test_kernel_series getAllTestVariants();
}

#endif // CLSPVTEST_TESTGREATERTHANOREQUALTO_KERNEL_HPP
