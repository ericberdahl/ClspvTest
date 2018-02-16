//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_READLOCALSIZE_KERNEL_HPP
#define CLSPVTEST_READLOCALSIZE_KERNEL_HPP

#include "clspv_utils.hpp"
#include "test_utils.hpp"
#include "util.hpp"

#include <vulkan/vulkan.hpp>

namespace readlocalsize_kernel {

    clspv_utils::kernel_invocation::execution_time_t
    invoke(const clspv_utils::kernel_module&    module,
           const clspv_utils::kernel&           kernel,
           const sample_info&                   info,
           vk::ArrayProxy<const vk::Sampler>    samplers,
           std::tuple<int, int, int>&           outLocalSizes);


    void test(const clspv_utils::kernel_module& module,
              const clspv_utils::kernel&        kernel,
              const sample_info&                info,
              vk::ArrayProxy<const vk::Sampler> samplers,
              test_utils::InvocationResultSet&  resultSet);

}


#endif //CLSPVTEST_READLOCALSIZE_KERNEL_HPP
