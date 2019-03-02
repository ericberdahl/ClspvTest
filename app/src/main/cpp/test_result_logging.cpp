//
// Created by Eric Berdahl on 4/26/18.
//

#include "test_result_logging.hpp"

#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

#include <boost/units/cmath.hpp>
#include <boost/units/io.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/prefixes.hpp>

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
        boost::units::quantity<boost::units::si::time> wallClockTime;
        boost::units::quantity<boost::units::si::time> executionTime;
        boost::units::quantity<boost::units::si::time> hostBarrierTime;
    };

    struct InvocationSummary {
        typedef decltype(test_utils::Evaluation::mMessages)::const_iterator   message_iterator;
        typedef iter_pair_range<message_iterator>   messages_t;

        ResultCounts                                    mCounts     = ResultCounts::null();
        execution_times                                 mTimes;
        boost::units::quantity<boost::units::si::time>  mTestTime;
        const std::string*                              mVariation  = nullptr;
        const std::string*                              mParameters = nullptr;
        unsigned int                                    mNumCorrect = 0;
        unsigned int                                    mNumErrors  = 0;
        messages_t                                      mMessages;
    };

    struct KernelSummary {
        std::string                     mEntryPoint;
        ResultCounts                    mCounts             = ResultCounts::null();
        std::vector<InvocationSummary>  mInvocationSummaries;
        const std::string*              mExceptionMessage   = nullptr;

        unsigned int                    mTimingIterations   = 0;
        execution_times                 mMeanTimes;
        execution_times                 mStdDeviationTimes;
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
        result.wallClockTime = ir.mExecutionTime.cpu_duration.count() * boost::units::si::seconds;
        result.executionTime = vulkan_utils::timestamp_delta(timestamps.host_barrier,
                                                             timestamps.execution,
                                                             info.physical_device_properties,
                                                             info.graphics_queue_family_properties);
        result.hostBarrierTime = vulkan_utils::timestamp_delta(timestamps.start,
                                                               timestamps.host_barrier,
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
        typedef boost::units::derived_dimension<boost::units::time_base_dimension, 2>::type time_variance_dimension;
        typedef boost::units::unit<time_variance_dimension, boost::units::si::system> time_variance;

        struct var_execution_times {
            boost::units::quantity<time_variance> wallClockTime;
            boost::units::quantity<time_variance> executionTime;
            boost::units::quantity<time_variance> hostBarrierTime;
        };

        std::vector<execution_times> times;
        times.reserve(resultSet.size());
        transform(resultSet.begin(), resultSet.end(), std::back_inserter(times),
                  [&info](const test_utils::InvocationTest::result &ir) {
                      return measureInvocationTime(info, ir.second);
                  });
        auto num_times = times.size();

        execution_times mean = accumulate(times.begin(), times.end(), execution_times(),
                                          [&info](execution_times accum, const execution_times &t) {
                                              accum.wallClockTime += t.wallClockTime;
                                              accum.executionTime += t.executionTime;
                                              accum.hostBarrierTime += t.hostBarrierTime;
                                              return accum;
                                          });
        mean.wallClockTime /= num_times;
        mean.executionTime /= num_times;
        mean.hostBarrierTime /= num_times;

        var_execution_times variance;
        if (num_times > 1) {
            variance = accumulate(times.begin(), times.end(), var_execution_times(),
                                  [mean](var_execution_times accum, const execution_times &t) {
                                      accum.wallClockTime += pow<2>(mean.wallClockTime - t.wallClockTime);
                                      accum.executionTime += pow<2>(mean.executionTime - t.executionTime);
                                      accum.hostBarrierTime += pow<2>(mean.hostBarrierTime - t.hostBarrierTime);
                                      return accum;
                                  });
            variance.wallClockTime /= (num_times - 1);
            variance.executionTime /= (num_times - 1);
            variance.hostBarrierTime /= (num_times - 1);
        }

        execution_times stdDeviation;
        stdDeviation.wallClockTime = sqrt(variance.wallClockTime);
        stdDeviation.executionTime = sqrt(variance.executionTime);
        stdDeviation.hostBarrierTime = sqrt(variance.hostBarrierTime);

        return std::make_pair(mean, stdDeviation);
    };

    InvocationSummary summarizeInvocation(const sample_info &info, const test_utils::InvocationTest::result& ir) {
        InvocationSummary result;
        result.mTimes = measureInvocationTime(info, ir.second);
        result.mTestTime = ir.second.mEvalTime.count() * boost::units::si::seconds;
        result.mNumCorrect = ir.second.mEvaluation.mNumCorrect;
        result.mNumErrors = ir.second.mEvaluation.mNumErrors;
        result.mMessages = std::make_pair(ir.second.mEvaluation.mMessages.begin(), ir.second.mEvaluation.mMessages.end());

        if (!ir.first->mVariation.empty()) result.mVariation = &ir.first->mVariation;
        if (!ir.second.mParameters.empty()) result.mParameters = &ir.second.mParameters;

        if (ir.second.mEvaluation.mSkipped) {
            result.mCounts = ResultCounts::skip();
        }
        else {
            // an invocation passes if it generates at least one correct value and no incorrect values
            result.mCounts = (ir.second.mEvaluation.mNumCorrect > 0 && ir.second.mEvaluation.mNumErrors == 0 ? ResultCounts::pass() : ResultCounts::fail());
        }

        return result;
    }

    KernelSummary summarizeKernel(const sample_info &info, const test_utils::KernelTest::result& kr) {
        KernelSummary result;
        result.mEntryPoint = kr.first->mEntryName;
        result.mTimingIterations = kr.first->mTimingIterations;

        if (!kr.second.mExceptionString.empty()) result.mExceptionMessage = &kr.second.mExceptionString;

        result.mInvocationSummaries.reserve(kr.second.mInvocationResults.size());
        std::transform(kr.second.mInvocationResults.begin(), kr.second.mInvocationResults.end(),
                       std::back_inserter(result.mInvocationSummaries),
                       std::bind(summarizeInvocation, std::cref(info), std::placeholders::_1));

        result.mCounts = std::accumulate(result.mInvocationSummaries.begin(), result.mInvocationSummaries.end(),
                                         ResultCounts::null(),
                                         [](ResultCounts r, const InvocationSummary& is) { return r + is.mCounts; });

        if (result.mTimingIterations > 0) {
            std::tie(result.mMeanTimes, result.mStdDeviationTimes) = computeSummaryStats(info, kr.second.mInvocationResults);
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

    std::string composeBasicInvocationSummary(const InvocationSummary& summary) {
        std::ostringstream os;
        os << (summary.mCounts.mSkip > 0 ? "SKIP" : (summary.mCounts.mFail > 0 ? "FAIL" : "PASS"));

        if (summary.mVariation) {
            os << " variation:" << *summary.mVariation;
        }

        if (summary.mParameters) {
            os << " parameters:" << *summary.mParameters;
        }

        return os.str();
    }

    void logInvocationSummary(const InvocationSummary& summary, unsigned int indent = 0) {
        std::ostringstream os;
        os << composeBasicInvocationSummary(summary);

        if (0 == summary.mCounts.mSkip) {
            os << boost::units::engineering_prefix
               << " correctValues:" << summary.mNumCorrect
               << " incorrectValues:" << summary.mNumErrors
               << " wallClockTime:" << summary.mTimes.wallClockTime
               << " resultEvalTime:" << summary.mTestTime
               << " executionTime:" << summary.mTimes.executionTime
               << " hostBarrierTime:" << summary.mTimes.hostBarrierTime;
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

        if (0 == summary.mTimingIterations) {
            std::for_each(summary.mInvocationSummaries.begin(), summary.mInvocationSummaries.end(),
                          std::bind(logInvocationSummary, std::placeholders::_1, indent + 1));
        } else {
            assert(!summary.mInvocationSummaries.empty());
            logInfo(composeBasicInvocationSummary(summary.mInvocationSummaries[0]), indent + 1);

            {
                std::ostringstream os;
                os << "NUMBER ITERATIONS = " << summary.mTimingIterations;
                logInfo(os.str(), indent + 1);
            }

            {
                std::ostringstream os;
                os << boost::units::engineering_prefix
                   << "AVERAGE "
                   << " wallClockTime:" << summary.mMeanTimes.wallClockTime
                   << " executionTime:" << summary.mMeanTimes.executionTime
                   << " hostBarrierTime:" << summary.mMeanTimes.hostBarrierTime;
                logInfo(os.str(), indent + 1);
            }

            if (summary.mInvocationSummaries.size() > 1) {
                std::ostringstream os;
                os << boost::units::engineering_prefix
                   << "STD_DEVIATION "
                   << " wallClockTime:" << summary.mStdDeviationTimes.wallClockTime
                   << " executionTime:" << summary.mStdDeviationTimes.executionTime
                   << " hostBarrierTime:" << summary.mStdDeviationTimes.hostBarrierTime;

                logInfo(os.str(), indent + 1);
            }
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