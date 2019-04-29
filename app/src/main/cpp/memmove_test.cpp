//
// Created by Eric Berdahl on 2019-04-23.
//

#include "memmove_test.hpp"

#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

#include "boost/accumulators/accumulators.hpp"
#include "boost/accumulators/statistics.hpp"
#include "boost/accumulators/statistics/mean.hpp"
#include "boost/accumulators/statistics/max.hpp"
#include "boost/accumulators/statistics/min.hpp"
#include "boost/accumulators/statistics/variance.hpp"
#include <boost/units/io.hpp>
#include <boost/units/systems/information.hpp>
#include <boost/units/systems/si.hpp>

#include <cassert>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace {

    template <typename T>
    struct statistics {
        typedef T   value_type;
        typedef T   variance_type;

        unsigned int    count       = 0;
        value_type      mean        = value_type();
        value_type      min         = value_type();
        value_type      max         = value_type();
        variance_type   variance    = variance_type();
    };

    template <>
    struct statistics<boost::units::quantity<boost::units::si::time>> {
        typedef boost::units::derived_dimension<boost::units::time_base_dimension, 2>::type time_variance_dimension;
        typedef boost::units::unit<time_variance_dimension, boost::units::si::system> time_variance;

        typedef boost::units::quantity<boost::units::si::time>  value_type;
        typedef boost::units::quantity<time_variance>           variance_type;

        unsigned int    count       = 0;
        value_type      mean        = value_type();
        value_type      min         = value_type();
        value_type      max         = value_type();
        variance_type   variance    = variance_type();
    };

    template <typename T>
    statistics<typename T::value_type> compute_stats(const T& container)
    {
        namespace ba = boost::accumulators;

        ba::accumulator_set<typename T::value_type,
                            ba::stats<ba::tag::count,
                                      ba::tag::mean,
                                      ba::tag::variance,
                                      ba::tag::max,
                                      ba::tag::min>> acc;
        acc = std::for_each(std::begin(container), std::end(container), acc);

        statistics<typename T::value_type> result;
        result.count = ba::count(acc);
        result.mean = ba::mean(acc);
        result.max = ba::max(acc);
        result.min = ba::min(acc);
        result.variance = ba::variance(acc);

        return result;
    }

    statistics<boost::units::quantity<boost::units::si::time>>
    compute_stats(const std::vector<test_utils::StopWatch::duration>& durations)
    {
        std::vector<double> values;
        values.reserve(durations.size());

        std::transform(durations.begin(), durations.end(),
                       std::back_inserter(values),
                       [](const test_utils::StopWatch::duration& d) { return d.count(); });
        const auto scalarStats = compute_stats(values);

        statistics<boost::units::quantity<boost::units::si::time>> results;
        results.count = scalarStats.count;
        results.mean = scalarStats.mean * boost::units::si::seconds;
        results.min = scalarStats.min * boost::units::si::seconds;
        results.max = scalarStats.max * boost::units::si::seconds;
        results.variance = scalarStats.variance * boost::units::si::seconds * boost::units::si::seconds;

        return results;
    }

    class AHBBuffer {
    public:
        AHBBuffer();

        AHBBuffer(AHardwareBuffer *buffer);

        AHBBuffer(const AHBBuffer& other) = delete;

        AHBBuffer(AHBBuffer&& other)
            : AHBBuffer()
        {
            swap(other);
        }

        ~AHBBuffer();

        void swap(AHBBuffer& other);

        void *get() { return mMappedPtr; }

    private:
        AHardwareBuffer*    mBuffer;
        void*               mMappedPtr;
    };

    AHBBuffer::AHBBuffer()
            : mBuffer(nullptr),
              mMappedPtr(nullptr)
    {
    }

    AHBBuffer::AHBBuffer(AHardwareBuffer *buffer)
            : AHBBuffer()
    {
        mBuffer = buffer;
        if (mBuffer)
        {
            //TODO lock the buffer
        }
    }

    AHBBuffer::~AHBBuffer()
    {
        // TODO unlock the buffer
        // TOOD delete the buffer
    }

    void AHBBuffer::swap(AHBBuffer& other)
    {
        using std::swap;

        swap(mBuffer, other.mBuffer);
        swap(mMappedPtr, other.mMappedPtr);
    }

    template <typename Fn>
    void runOneTest(const std::string& label,
                    std::size_t bufferSize,
                    unsigned int numIterations,
                    Fn testFn)
    {
        const auto stats = compute_stats(testFn(bufferSize, numIterations));

        const boost::units::quantity<boost::units::information::info> bufferBytes = bufferSize * boost::units::information::bytes;

        std::ostringstream os;
        os << boost::units::engineering_prefix
           << label
           << " bufferSize:" << bufferBytes
           << " count:" << stats.count
           << " mean:" << stats.mean
           << " rate:" << bufferBytes/stats.mean
           << " variance" << stats.variance
           << " max:" << stats.max
           << " min:" << stats.min;

        LOGI("%s", os.str().c_str());
    }
}

namespace memmove_test{

    std::vector<test_utils::StopWatch::duration> timeSystem2System(std::size_t bufferSize,
                                                                   unsigned int iterations) {
        std::vector<std::uint8_t>   source(bufferSize);
        std::vector<std::uint8_t>   destination(bufferSize);

        return timeTransfer(source.data(), destination.data(), bufferSize, iterations);
    }

    std::vector<test_utils::StopWatch::duration> timeSystem2VkDeviceMemory(vk::Device device,
                                                                           std::size_t bufferSize,
                                                                           const vk::PhysicalDeviceMemoryProperties &memProps,
                                                                           vk::BufferUsageFlags usageFlags,
                                                                           unsigned int iterations) {
        std::vector<std::uint8_t>   source(bufferSize);
        vulkan_utils::buffer        destBuffer(device, memProps, bufferSize, usageFlags);

        auto destination = destBuffer.map();

        return timeTransfer(source.data(), destination.get(), bufferSize, iterations);
    }

    std::vector<test_utils::StopWatch::duration> timeTransfer(const void* source,
                                                              void* destination,
                                                              std::size_t bufferSize,
                                                              unsigned int iterations) {
        std::vector<test_utils::StopWatch::duration> results;
        results.reserve(iterations);

        test_utils::StopWatch stopWatch;
        for (unsigned int i = iterations; i > 0; --i) {
            stopWatch.restart();
            std::memmove(destination, source, bufferSize);
            results.push_back(stopWatch.getSplitTime());
        }

        return results;
    }

    void runAllTests(const sample_info& info)
    {
        const std::size_t   bufferSizes[] = {
                3840 * 2160 * 4,
                1920 * 1080 * 4,
                1280 *  720 * 4,
        };

        // eStorageBuffer vs eUniformBuffer
        // eTransferSrc
        // eTransferDest
        const vk::BufferUsageFlags usageFlags[] = {
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        };

        const std::size_t   bufferSize      = 3840 * 2160 * 4;
        const unsigned int  numIterations   = 10;

        const vk::Device                            device      = *info.device;
        const vk::PhysicalDeviceMemoryProperties    memProps    = info.gpu.getMemoryProperties();

        for (auto size : bufferSizes)
        {
            runOneTest("system-system", size, numIterations, timeSystem2System);

            for (auto usage : usageFlags)
            {
                const auto timeSystem2Vulkan = [&device, &memProps, usage](std::size_t bufferSize,
                                                                           unsigned int numIterations) {
                    return timeSystem2VkDeviceMemory(device, bufferSize, memProps,
                                                     usage,
                                                     numIterations);
                };

                runOneTest("system-vulkan(" + vk::to_string(usage) + ")", size, numIterations, timeSystem2Vulkan);
            }
        }
    }
}