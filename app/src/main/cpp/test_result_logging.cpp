//
// Created by Eric Berdahl on 4/26/18.
//

#include "test_result_logging.hpp"

#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <sstream>
#include <utility>

namespace {
    template <typename Iter>
    struct iter_pair_range : std::pair<Iter,Iter> {
        iter_pair_range()
                : std::pair<Iter,Iter>()
        {}

        iter_pair_range(std::pair<Iter,Iter> const& x)
                : std::pair<Iter,Iter>(x)
        {}
        Iter begin() const {return this->first;}
        Iter end()   const {return this->second;}
    };


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

    struct execution_times {
        double wallClockTime_s      = 0.0;
        double executionTime_ns     = 0.0;
        double hostBarrierTime_ns   = 0.0;
        double gpuBarrierTime_ns    = 0.0;
    };

    struct InvocationSummary {
        typedef decltype(test_utils::InvocationResult::mMessages)::const_iterator   message_iterator;
        typedef iter_pair_range<message_iterator>   messages_t;

        ResultCounts        mCounts     = ResultCounts::null();
        execution_times     mTimes;
        double              mTestTime_s = 0.0;
        const std::string*  mVariation  = nullptr;
        const std::string*  mParameters = nullptr;
        unsigned int        mNumCorrect = 0;
        unsigned int        mNumErrors  = 0;
        messages_t          mMessages;
    };

    struct KernelSummary {
        std::string                     mEntryPoint;
        ResultCounts                    mCounts             = ResultCounts::null();
        std::vector<InvocationSummary>  mInvocationSummaries;
        const std::string*              mExceptionMessage   = nullptr;

        unsigned int                    mIterations         = 0;
        execution_times                 mMeanTimes;
        execution_times                 mVarianceTimes;
    };

    struct ModuleSummary {
        typedef decltype(test_utils::ModuleResult::mUntestedEntryPoints)::const_iterator   untested_entry_iterator;
        typedef iter_pair_range<untested_entry_iterator>   untested_entries_t;

        std::string                 mName;
        ResultCounts                mCounts             = ResultCounts::null();
        std::vector<KernelSummary>  mKernelSummaries;
        const std::string*          mExceptionMessage   = nullptr;
        untested_entries_t          mUntestedEntries;
    };

    struct ManifestSummary {
        std::vector<ModuleSummary>  mModuleSummaries;
        ResultCounts                mCounts = ResultCounts::null();
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

    void logInfo(const std::string& s, unsigned int indentLevel) {
        LOGI("%*s%s", indentLevel*3, "", s.c_str());
    }

    void logDebug(const std::string& s, unsigned int indentLevel) {
        LOGD("%*s%s", indentLevel*3, "", s.c_str());
    }

    std::pair<execution_times, execution_times>
    computeSummaryStats(const sample_info &info, const test_utils::KernelResult::results &resultSet) {
        std::vector<execution_times> times;
        times.reserve(resultSet.size());
        transform(resultSet.begin(), resultSet.end(), std::back_inserter(times),
                  [&info](const test_utils::InvocationTest::result &ir) {
                      return measureInvocationTime(info, ir.second);
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

        execution_times variance;
        if (num_times > 1) {
            variance = accumulate(times.begin(), times.end(), execution_times(),
                                  [mean](execution_times accum, const execution_times &t) {
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
        }

        return std::make_pair(mean, variance);
    };

    InvocationSummary summarizeInvocation(const sample_info &info, const test_utils::InvocationTest::result& ir) {
        InvocationSummary result;
        result.mTimes = measureInvocationTime(info, ir.second);
        result.mTestTime_s = ir.second.mTestTime.count();
        result.mNumCorrect = ir.second.mNumCorrect;
        result.mNumErrors = ir.second.mNumErrors;
        result.mMessages = std::make_pair(ir.second.mMessages.begin(), ir.second.mMessages.end());

        if (!ir.first->mVariation.empty()) result.mVariation = &ir.first->mVariation;
        if (!ir.second.mParameters.empty()) result.mParameters = &ir.second.mParameters;

        if (ir.second.mSkipped) {
            result.mCounts = ResultCounts::skip();
        }
        else {
            // an invocation passes if it generates at least one correct value and no incorrect values
            result.mCounts = (ir.second.mNumCorrect > 0 && ir.second.mNumErrors == 0 ? ResultCounts::pass() : ResultCounts::fail());
        }

        return result;
    }

    KernelSummary summarizeKernel(const sample_info &info, const test_utils::KernelTest::result& kr) {
        KernelSummary result;
        result.mEntryPoint = kr.first->mEntryName;
        result.mIterations = kr.first->mIterations;

        if (!kr.second.mExceptionString.empty()) result.mExceptionMessage = &kr.second.mExceptionString;

        result.mInvocationSummaries.reserve(kr.second.mInvocationResults.size());
        std::transform(kr.second.mInvocationResults.begin(), kr.second.mInvocationResults.end(),
                       std::back_inserter(result.mInvocationSummaries),
                       std::bind(summarizeInvocation, std::cref(info), std::placeholders::_1));

        result.mCounts = std::accumulate(result.mInvocationSummaries.begin(), result.mInvocationSummaries.end(),
                                         ResultCounts::null(),
                                         [](ResultCounts r, const InvocationSummary& is) { return r + is.mCounts; });

        if (result.mIterations > 1) {
            std::tie(result.mMeanTimes, result.mVarianceTimes) = computeSummaryStats(info, kr.second.mInvocationResults);
        }

        return result;
    }

    ModuleSummary summarizeModule(const sample_info &info, const test_utils::ModuleTest::result& mr) {
        ModuleSummary result;
        result.mName = mr.first->mName;
        result.mUntestedEntries = std::make_pair(mr.second.mUntestedEntryPoints.begin(), mr.second.mUntestedEntryPoints.end());

        if (!mr.second.mExceptionString.empty()) result.mExceptionMessage = &mr.second.mExceptionString;

        result.mKernelSummaries.reserve(mr.second.mKernelResults.size());
        std::transform(mr.second.mKernelResults.begin(), mr.second.mKernelResults.end(),
                       std::back_inserter(result.mKernelSummaries),
                       std::bind(summarizeKernel, std::cref(info), std::placeholders::_1));

        result.mCounts = std::accumulate(result.mKernelSummaries.begin(), result.mKernelSummaries.end(),
                                         ResultCounts::null(),
                                         [](ResultCounts r, const KernelSummary& ks) { return r + ks.mCounts; });

        return result;
    }

    ManifestSummary summarizeManifest(const sample_info& info, const test_manifest::results& manifestResults) {
        ManifestSummary result;

        result.mModuleSummaries.reserve(manifestResults.size());
        std::transform(manifestResults.begin(), manifestResults.end(),
                       std::back_inserter(result.mModuleSummaries),
                       std::bind(summarizeModule, std::cref(info), std::placeholders::_1));

        result.mCounts = std::accumulate(result.mModuleSummaries.begin(), result.mModuleSummaries.end(),
                                         ResultCounts::null(),
                                         [](ResultCounts r, const ModuleSummary& ms) {
                                             return r + ms.mCounts;
                                         });

        return result;
    }

    void logInvocationSummary(const InvocationSummary& summary, unsigned int indent = 0) {
        std::ostringstream os;
        os << (summary.mCounts.mSkip > 0 ? "SKIP" : (summary.mCounts.mFail > 0 ? "FAIL" : "PASS"));

        if (summary.mVariation) {
            os << " variation:" << *summary.mVariation;
        }

        if (summary.mParameters) {
            os << " parameters:" << *summary.mParameters;
        }

        if (0 == summary.mCounts.mSkip) {
            os << " correctValues:" << summary.mNumCorrect
               << " incorrectValues:" << summary.mNumErrors
               << " wallClockTime:" << summary.mTimes.wallClockTime_s * 1000.0f << "ms"
               << " resultEvalTime:" << summary.mTestTime_s << "s"
               << " executionTime:" << summary.mTimes.executionTime_ns / 1000.0f << "µs"
               << " hostBarrierTime:" << summary.mTimes.hostBarrierTime_ns / 1000.0f << "µs"
               << " gpuBarrierTime:" << summary.mTimes.gpuBarrierTime_ns / 1000.0f << "µs";
        }

        logInfo(os.str(), indent);

        for (auto& err : summary.mMessages) {
            logDebug(err, indent + 1);
        }
    }

    void logKernelSummary(const KernelSummary& summary, unsigned int indent = 0) {
        {
            std::ostringstream os;
            os << "Kernel:" << summary.mEntryPoint << " " << summary.mCounts;
            logInfo(os.str(), indent);
        }
        if (summary.mExceptionMessage) {
            std::ostringstream os;
            os << "exception: " << *summary.mExceptionMessage;
            logInfo(os.str(), indent + 1);
        }

        std::for_each(summary.mInvocationSummaries.begin(), summary.mInvocationSummaries.end(),
                      std::bind(logInvocationSummary, std::placeholders::_1, indent + 1));

        if (summary.mIterations > 1) {
            std::ostringstream os;
            os << "AVERAGE "
               << " wallClockTime:" << summary.mMeanTimes.wallClockTime_s * 1000.0f << "ms"
               << " executionTime:" << summary.mMeanTimes.executionTime_ns / 1000.0f << "µs"
               << " hostBarrierTime:" << summary.mMeanTimes.hostBarrierTime_ns / 1000.0f << "µs"
               << " gpuBarrierTime:" << summary.mMeanTimes.gpuBarrierTime_ns / 1000.0f << "µs";
            logInfo(os.str(), indent + 1);
        }

        if (summary.mIterations > 1 && summary.mInvocationSummaries.size() > 1) {
            std::ostringstream os;
            os << "STD_DEVIATION "
               << " wallClockTime:" << sqrt(summary.mVarianceTimes.wallClockTime_s) * 1000.0f << "ms"
               << " executionTime:" << sqrt(summary.mVarianceTimes.executionTime_ns) / 1000.0f << "µs"
               << " hostBarrierTime:" << sqrt(summary.mVarianceTimes.hostBarrierTime_ns) / 1000.0f << "µs"
               << " gpuBarrierTime:" << sqrt(summary.mVarianceTimes.gpuBarrierTime_ns) / 1000.0f << "µs";

            logInfo(os.str(), indent + 1);
        }
    }

    void logModuleSummary(const ModuleSummary& summary, unsigned int indent = 0) {
        {
            std::ostringstream os;
            os << "Module:" << summary.mName << " " << summary.mCounts;
            logInfo(os.str(), indent);
        }

        if (summary.mExceptionMessage) {
            std::ostringstream os;
            os << "exception:" << *summary.mExceptionMessage;
            logInfo(os.str(), indent + 1);
        }

        for (auto& untested : summary.mUntestedEntries) {
            std::ostringstream os;
            os << "MISSED " << untested;
            logInfo(os.str(), indent + 1);
        }

        std::for_each(summary.mKernelSummaries.begin(), summary.mKernelSummaries.end(),
                      std::bind(logKernelSummary, std::placeholders::_1, indent + 1));
    }

    void logManifestSummary(const ManifestSummary& summary, unsigned int indent = 0) {
        auto longestName = std::max_element(summary.mModuleSummaries.begin(), summary.mModuleSummaries.end(),
                                            [](const ModuleSummary& lhs, const ModuleSummary& rhs) {
                                               return lhs.mName.size() < rhs.mName.size();
                                            });
        const std::size_t width = longestName->mName.size();

        std::vector<std::string> moduleCountStrings;
        std::string overallCountString;

        for (auto& s : summary.mModuleSummaries) {
            std::ostringstream os;
            os << std::setw(width) << s.mName << ": " << s.mCounts;
            moduleCountStrings.push_back(os.str());
        }

        {
            std::ostringstream os;
            os << "Overall Summary " << summary.mCounts;
            overallCountString = os.str();
        }

        logInfo("Module Summaries", indent);
        std::for_each(moduleCountStrings.begin(), moduleCountStrings.end(),
                      std::bind(logInfo, std::placeholders::_1, indent + 1));
        logInfo(overallCountString, indent);

        std::for_each(summary.mModuleSummaries.begin(), summary.mModuleSummaries.end(),
                      std::bind(logModuleSummary, std::placeholders::_1, indent));

        logInfo("Module Summaries", indent);
        std::for_each(moduleCountStrings.begin(), moduleCountStrings.end(),
                      std::bind(logInfo, std::placeholders::_1, indent + 1));
        logInfo(overallCountString, indent);
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
        logInfo(os.str(), 0);
    }

    void logResults(const sample_info &info, const test_utils::InvocationTest::result &ir) {
        const InvocationSummary summary = summarizeInvocation(info, ir);
        logInvocationSummary(summary);
    }

    void logResults(const sample_info &info, const test_utils::KernelTest::result &kr) {
        const KernelSummary summary = summarizeKernel(info, kr);
        logKernelSummary(summary);
    }

    void logResults(const sample_info &info, const test_utils::ModuleTest::result &mr) {
        const ModuleSummary summary = summarizeModule(info, mr);
        logModuleSummary(summary);
    }

    void logResults(const sample_info &info, const test_manifest::results &moduleResultSet) {
        logPhysicalDeviceInfo(info);

        const ManifestSummary summary = summarizeManifest(info, moduleResultSet);
        logManifestSummary(summary);
    }

}