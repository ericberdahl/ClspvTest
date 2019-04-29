//
// Created by Eric Berdahl on 2019-04-23.
//

#ifndef CLSPVTEST_MEMMOVETEST_HPP
#define CLSPVTEST_MEMMOVETEST_HPP

#include "test_utils.hpp"
#include "util.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <vector>

namespace memmove_test {

    std::vector<test_utils::StopWatch::duration> timeSystem2System(std::size_t bufferSize,
                                                                   unsigned int iterations);

    std::vector<test_utils::StopWatch::duration> timeSystem2VkDeviceMemory(vk::Device device,
                                                                           std::size_t bufferSize,
                                                                           const vk::PhysicalDeviceMemoryProperties &memProps,
                                                                           vk::BufferUsageFlags usageFlags,
                                                                           unsigned int iterations);

    std::vector<test_utils::StopWatch::duration> timeTransfer(const void* source,
                                                              void* destination,
                                                              std::size_t bufferSize,
                                                              unsigned int iterations);

    void runAllTests(const sample_info& info);
}

#endif //CLSPVTEST_MEMMOVETEST_HPP
