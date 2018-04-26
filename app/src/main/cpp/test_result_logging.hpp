//
// Created by Eric Berdahl on 4/26/18.
//

#ifndef CLSPVTEST_TEST_RESULT_LOGGING_HPP
#define CLSPVTEST_TEST_RESULT_LOGGING_HPP

#include "test_utils.hpp"

struct sample_info;

namespace test_result_logging {
    void logPhysicalDeviceInfo(const sample_info &info);

    void logResults(const sample_info &info, const test_utils::InvocationResult &ir);

    void logResults(const sample_info &info, const test_utils::KernelResult &kr);

    void logResults(const sample_info &info, const test_utils::ModuleResult &mr);

    void logResults(const sample_info &info, const test_utils::ModuleResultSet &moduleResultSet);
}

#endif //CLSPVTEST_TEST_RESULT_LOGGING_HPP
