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
    struct ResultCounts {
        static ResultCounts null() { return ResultCounts(0, 0, 0); }
        static ResultCounts pass() { return ResultCounts(1, 0, 0); }
        static ResultCounts fail() { return ResultCounts(0, 1, 0); }
        static ResultCounts skip() { return ResultCounts(0, 0, 1); }

        ResultCounts(unsigned int pass, unsigned int fail, unsigned int skip)
                : mPass(pass), mFail(fail), mSkip(skip) {}

        ResultCounts&   operator+=(const ResultCounts& addend) {
            mPass += addend.mPass;
            mFail += addend.mFail;
            mSkip += addend.mSkip;
            return *this;
        }

        unsigned int    mPass;
        unsigned int    mFail;
        unsigned int    mSkip;
    };

    ResultCounts operator+(ResultCounts lhs, const ResultCounts& rhs) {
        lhs += rhs;
        return lhs;
    }

    std::ostream& operator<<(std::ostream& os, const ResultCounts& results) {
        os << "pass:" << results.mPass
           << " fail:" << results.mFail
           << " skipped:" << results.mSkip;
        return os;
    }

    ResultCounts countResults(const test_utils::InvocationResult &ir) {
        if (ir.mSkipped) return ResultCounts::skip();

        // an invocation passes if it generates at least one correct value and no incorrect values
        return (ir.mNumCorrect > 0 && ir.mNumErrors == 0 ? ResultCounts::pass() : ResultCounts::fail());
    };

    ResultCounts countResults(const test_utils::KernelResult &kr) {
        if (kr.mSkipped) {
            assert(kr.mInvocations.empty());
            return ResultCounts::skip();
        }

        // a kernel's results are the aggregate sum of its invocations
        return std::accumulate(kr.mInvocations.begin(), kr.mInvocations.end(),
                               kr.mCompiledCorrectly ? ResultCounts::pass() : ResultCounts::fail(),
                               [](ResultCounts r, const test_utils::InvocationResult &ir) {
                                   return r + countResults(ir);
                               });
    };

    ResultCounts countResults(const test_utils::ModuleResult &mr) {
        // a module's results are the aggregate sum of its kernels, combined with the result of its own
        // loading (i.e. whether it loaded correctly or not)
        return std::accumulate(mr.mKernels.begin(), mr.mKernels.end(),
                               mr.mLoadedCorrectly ? ResultCounts::pass() : ResultCounts::fail(),
                               [](ResultCounts r, const test_utils::KernelResult &kr) {
                                   return r + countResults(kr);
                               });
    };

    ResultCounts countResults(const test_utils::ModuleResultSet &moduleResultSet) {
        return std::accumulate(moduleResultSet.begin(), moduleResultSet.end(),
                               ResultCounts::null(),
                               [](ResultCounts r, const test_utils::ModuleResult &mr) {
                                   return r + countResults(mr);
                               });
    };

    const char* resultString(bool passed, bool skipped = false) {
        if (skipped) return "SKIP";
        return (passed ? "PASS" : "FAIL");
    }

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
        os << resultString(ir.mNumCorrect > 0 && ir.mNumErrors == 0, ir.mSkipped);

        if (!ir.mVariation.empty()) {
            os << " variation:" << ir.mVariation << "";
        }

        if (!ir.mSkipped) {
            os << " correctValues:" << ir.mNumCorrect
               << " incorrectValues:" << ir.mNumErrors
               << " wallClockTime:" << times.wallClockTime_s * 1000.0f << "ms"
               << " executionTime:" << times.executionTime_ns / 1000.0f << "µs"
               << " hostBarrierTime:" << times.hostBarrierTime_ns / 1000.0f << "µs"
               << " gpuBarrierTime:" << times.gpuBarrierTime_ns / 1000.0f << "µs";
        }

        LOGI("      %s", os.str().c_str());

        for (auto err : ir.mMessages) {
            LOGD("         %s", err.c_str());
        }
    }

    void logResults(const sample_info &info, const test_utils::KernelResult &kr) {
        auto results = countResults(kr);

        {
            std::ostringstream os;
            os << "Kernel:" << kr.mEntryName << " " << results;
            LOGI("   %s", os.str().c_str());
        }
        {
            std::ostringstream os;

            if (kr.mSkipped) {
                os << "SKIP";
            }
            else {
                os << resultString(kr.mCompiledCorrectly) << " compilation";
            }
            if (!kr.mExceptionString.empty()) {
                os << " exception:" << kr.mExceptionString;
            }
            LOGI("      %s", os.str().c_str());
        }

        for (auto ir : kr.mInvocations) {
            logResults(info, ir);
        }

        if (kr.mIterations > 1) {
            logSummaryStats(info, kr.mInvocations);
        }
    }

    void logResults(const sample_info &info, const test_utils::ModuleResult &mr) {
        auto results = countResults(mr);

        {
            std::ostringstream os;
            os << "Module:" << mr.mModuleName << " " << results;
            LOGI("%s", os.str().c_str());
        }

        {
            std::ostringstream os;
            os << (mr.mLoadedCorrectly ? "PASS" : "FAIL") << " moduleLoading";
            if (!mr.mExceptionString.empty()) {
                os << " exception:" << mr.mExceptionString;
            }
            LOGI("   %s", os.str().c_str());
        }

        for (auto kr : mr.mKernels) {
            logResults(info, kr);
        }
    }

    void logResults(const sample_info &info, const test_utils::ModuleResultSet &moduleResultSet) {
        logPhysicalDeviceInfo(info);

        auto results = countResults(moduleResultSet);

        std::ostringstream os;
        os << "Overall Summary " << results;
        LOGI("%s", os.str().c_str());

        for (auto mr : moduleResultSet) {
            logResults(info, mr);
        }

        LOGI("%s", os.str().c_str());
    }

}