//
// Created by Eric Berdahl on 4/26/18.
//

#include "test_result_logging.hpp"

#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <sstream>
#include <utility>

namespace {
    std::pair<unsigned int, unsigned int> countResults(const test_utils::InvocationResult &ir) {
        // an invocation passes if it generates at least one correct value and no incorrect values
        return (ir.mNumCorrect > 0 && ir.mNumErrors == 0 ? std::make_pair(1, 0) : std::make_pair(0,
                                                                                                 1));
    };

    std::pair<unsigned int, unsigned int> countResults(const test_utils::KernelResult &kr) {
        // a kernel's results are the aggregate sum of its invocations
        return std::accumulate(kr.mInvocations.begin(), kr.mInvocations.end(),
                               std::make_pair(0, 0),
                               [](std::pair<unsigned int, unsigned int> r,
                                  const test_utils::InvocationResult &ir) {
                                   auto addend = countResults(ir);
                                   r.first += addend.first;
                                   r.second += addend.second;
                                   return r;
                               });
    };

    std::pair<unsigned int, unsigned int> countResults(const test_utils::ModuleResult &mr) {
        // a module's results are the aggregate sum of its kernels, combined with the result of its own
        // loading (i.e. whether it loaded correctly or not)
        return std::accumulate(mr.mKernels.begin(), mr.mKernels.end(),
                               mr.mLoadedCorrectly ? std::make_pair(1, 0) : std::make_pair(0, 1),
                               [](std::pair<unsigned int, unsigned int> r,
                                  const test_utils::KernelResult &kr) {
                                   auto addend = countResults(kr);
                                   r.first += addend.first;
                                   r.second += addend.second;
                                   return r;
                               });
    };

    std::pair<unsigned int, unsigned int>
    countResults(const test_utils::ModuleResultSet &moduleResultSet) {
        return std::accumulate(moduleResultSet.begin(), moduleResultSet.end(),
                               std::make_pair(0, 0),
                               [](std::pair<unsigned int, unsigned int> r,
                                  const test_utils::ModuleResult &mr) {
                                   auto addend = countResults(mr);
                                   r.first += addend.first;
                                   r.second += addend.second;
                                   return r;
                               });
    };

    struct execution_times {
        double wallClockTime_s = 0.0;
        double executionTime_ns = 0.0;
        double hostBarrierTime_ns = 0.0;
        double gpuBarrierTime_ns = 0.0;
    };

    execution_times
    measureInvocationTime(const sample_info &info, const test_utils::InvocationResult &ir) {
        auto &timestamps = ir.mExecutionTime.timestamps;

        execution_times result;
        result.wallClockTime_s = ir.mExecutionTime.cpu_duration.count();
        result.executionTime_ns = vulkan_utils::timestamp_delta_ns(timestamps.host_barrier,
                                                                   timestamps.execution,
                                                                   info.physical_device_properties,
                                                                   info.graphics_queue_family_properties);
        result.hostBarrierTime_ns = vulkan_utils::timestamp_delta_ns(timestamps.start,
                                                                     timestamps.host_barrier,
                                                                     info.physical_device_properties,
                                                                     info.graphics_queue_family_properties);
        result.gpuBarrierTime_ns = vulkan_utils::timestamp_delta_ns(timestamps.execution,
                                                                    timestamps.gpu_barrier,
                                                                    info.physical_device_properties,
                                                                    info.graphics_queue_family_properties);

        return result;
    }

    void
    logSummaryStats(const sample_info &info, const test_utils::InvocationResultSet &resultSet) {
        std::vector<execution_times> times;
        times.reserve(resultSet.size());
        transform(resultSet.begin(), resultSet.end(), std::back_inserter(times),
                  [&info](const test_utils::InvocationResult &ir) {
                      return measureInvocationTime(info, ir);
                  });
        auto num_times = times.size();

        execution_times mean = accumulate(times.begin(), times.end(), execution_times(),
                                          [&info](execution_times accum, const execution_times &t) {
                                              accum.wallClockTime_s += t.wallClockTime_s;
                                              accum.executionTime_ns += t.executionTime_ns;
                                              accum.hostBarrierTime_ns += t.hostBarrierTime_ns;
                                              accum.gpuBarrierTime_ns += t.gpuBarrierTime_ns;
                                              return accum;
                                          });
        mean.wallClockTime_s /= num_times;
        mean.executionTime_ns /= num_times;
        mean.hostBarrierTime_ns /= num_times;
        mean.gpuBarrierTime_ns /= num_times;

        std::ostringstream os;
        os << "AVERAGE "
           << " wallClockTime:" << mean.wallClockTime_s * 1000.0f << "ms"
           << " executionTime:" << mean.executionTime_ns / 1000.0f << "µs"
           << " hostBarrierTime:" << mean.hostBarrierTime_ns / 1000.0f << "µs"
           << " gpuBarrierTime:" << mean.gpuBarrierTime_ns / 1000.0f << "µs";

        LOGI("      %s", os.str().c_str());

        if (num_times > 1) {
            execution_times variance = accumulate(times.begin(), times.end(), execution_times(),
                                                  [mean](execution_times accum,
                                                         const execution_times &t) {
                                                      accum.wallClockTime_s += pow(
                                                              mean.wallClockTime_s -
                                                              t.wallClockTime_s, 2);
                                                      accum.executionTime_ns += pow(
                                                              mean.executionTime_ns -
                                                              t.executionTime_ns, 2);
                                                      accum.hostBarrierTime_ns += pow(
                                                              mean.hostBarrierTime_ns -
                                                              t.hostBarrierTime_ns, 2);
                                                      accum.gpuBarrierTime_ns += pow(
                                                              mean.gpuBarrierTime_ns -
                                                              t.gpuBarrierTime_ns, 2);
                                                      return accum;
                                                  });
            variance.wallClockTime_s /= (num_times - 1);
            variance.executionTime_ns /= (num_times - 1);
            variance.hostBarrierTime_ns /= (num_times - 1);
            variance.gpuBarrierTime_ns /= (num_times - 1);

            os.clear();
            os.str("");
            os << "STD_DEVIATION "
               << " wallClockTime:" << sqrt(variance.wallClockTime_s) * 1000.0f << "ms"
               << " executionTime:" << sqrt(variance.executionTime_ns) / 1000.0f << "µs"
               << " hostBarrierTime:" << sqrt(variance.hostBarrierTime_ns) / 1000.0f << "µs"
               << " gpuBarrierTime:" << sqrt(variance.gpuBarrierTime_ns) / 1000.0f << "µs";

            LOGI("      %s", os.str().c_str());

        }
    }
}

namespace test_result_logging {

    void logPhysicalDeviceInfo(const sample_info &info) {
        const vk::PhysicalDeviceProperties props = info.gpu.getProperties();
        std::ostringstream os;
        os << "PhysicalDevice {" << std::endl
           << "   apiVersion:" << props.apiVersion << std::endl
           << "   driverVersion:" << props.driverVersion << std::endl
           << "   vendorID:" << props.vendorID << std::endl
           << "   deviceID:" << props.deviceID << std::endl
           << "   deviceName:" << props.deviceName << std::endl
           << "}";
        LOGI("%s", os.str().c_str());
    }

    void logResults(const sample_info &info, const test_utils::InvocationResult &ir) {
        const execution_times times = measureInvocationTime(info, ir);

        std::ostringstream os;
        os << (ir.mNumCorrect > 0 && ir.mNumErrors == 0 ? "PASS" : "FAIL");

        if (!ir.mVariation.empty()) {
            os << " variation:" << ir.mVariation << "";
        }

        os << " correctValues:" << ir.mNumCorrect
           << " incorrectValues:" << ir.mNumErrors
           << " wallClockTime:" << times.wallClockTime_s * 1000.0f << "ms"
           << " executionTime:" << times.executionTime_ns / 1000.0f << "µs"
           << " hostBarrierTime:" << times.hostBarrierTime_ns / 1000.0f << "µs"
           << " gpuBarrierTime:" << times.gpuBarrierTime_ns / 1000.0f << "µs";

        LOGI("      %s", os.str().c_str());

        for (auto err : ir.mMessages) {
            LOGD("         %s", err.c_str());
        }
    }

    void logResults(const sample_info &info, const test_utils::KernelResult &kr) {
        std::ostringstream os;

        os << "Kernel:" << kr.mEntryName;
        if (kr.mSkipped) {
            os << " SKIPPED";
        } else if (!kr.mCompiledCorrectly) {
            os << " COMPILE-FAILURE";
        }
        if (!kr.mExceptionString.empty()) {
            os << " " << kr.mExceptionString;
        }
        LOGI("   %s", os.str().c_str());

        for (auto ir : kr.mInvocations) {
            logResults(info, ir);
        }

        if (kr.mIterations > 1) {
            logSummaryStats(info, kr.mInvocations);
        }
    }

    void logResults(const sample_info &info, const test_utils::ModuleResult &mr) {
        std::ostringstream os;
        os << "Module:" << mr.mModuleName;
        if (!mr.mExceptionString.empty()) {
            os << " loadException:" << mr.mExceptionString;
        }
        LOGI("%s", os.str().c_str());

        for (auto kr : mr.mKernels) {
            logResults(info, kr);
        }
    }

    void logResults(const sample_info &info, const test_utils::ModuleResultSet &moduleResultSet) {
        logPhysicalDeviceInfo(info);

        for (auto mr : moduleResultSet) {
            logResults(info, mr);
        }

        auto results = countResults(moduleResultSet);

        std::ostringstream os;
        os << "Overall Summary"
           << " pass:" << results.first << " fail:" << results.second;
        LOGI("%s", os.str().c_str());
    }

}