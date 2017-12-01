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

    std::tuple<int, int, int> invoke(const clspv_utils::kernel_module&  module,
                                     const clspv_utils::kernel&         kernel,
                                     const sample_info&                 info,
                                     vk::ArrayProxy<const vk::Sampler>  samplers);


    test_utils::Results test(const clspv_utils::kernel_module&  module,
                             const clspv_utils::kernel&         kernel,
                             const sample_info&                 info,
                             vk::ArrayProxy<const vk::Sampler>  samplers,
                             const test_utils::options&         opts);

}


#endif //CLSPVTEST_READLOCALSIZE_KERNEL_HPP
