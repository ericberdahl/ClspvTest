//
// Created by Eric Berdahl on 10/31/17.
//

#include "test_utils.hpp"

#include "clspv_utils/interface.hpp"
#include "clspv_utils/kernel.hpp"
#include "clspv_utils/kernel_module.hpp"

#include "crlf_savvy.hpp"
#include "file_utils.hpp"

namespace {
    using namespace test_utils;

    std::string current_exception_to_string() {
        std::string result;

        try {
            throw;
        }
        catch (const vk::SystemError &e) {
            std::ostringstream os;
            os << "vk::SystemError : " << e.code() << " (" << e.code().message() << ')';
            result = os.str();
        }
        catch (const std::system_error &e) {
            std::ostringstream os;
            os << "std::system_error : " << e.code() << " (" << e.code().message() << ')';
            result = os.str();
        }
        catch (const std::exception &e) {
            std::ostringstream os;
            os << "std::exception : " << e.what();
            result = os.str();
        }
        catch (...) {
            result = "unknown exception";
        }

        return result;
    }

    InvocationResult null_invocation_test(clspv_utils::kernel &kernel,
                                          const std::vector<std::string> &args,
                                          bool verbose)
    {
        InvocationResult result;
        result.mNumCorrect = 1;
        result.mMessages.push_back("kernel compiled but intentionally not invoked");
        return result;
    }

    InvocationResult failTestFn(clspv_utils::kernel&             kernel,
                                const std::vector<std::string>&  args,
                                bool                             verbose)
    {
        InvocationResult result;
        result.mMessages.push_back("kernel failed to compile");
        return result;
    }

}

namespace test_utils {

    KernelTest::result test_kernel(clspv_utils::kernel_module&    module,
                                   const KernelTest&              kernelTest) {
        KernelTest::result result;
        result.first = &kernelTest;
        result.second.mSkipped = false;

        clspv_utils::kernel kernel;

		try {
	        kernel = module.createKernel(kernelTest.mEntryName, kernelTest.mWorkgroupSize);
            result.second.mCompiledCorrectly = true;
		}
        catch (...) {
            result.second.mExceptionString = current_exception_to_string();
        }

        if (!kernelTest.mInvocationTests.empty()) {
            try {
                for (unsigned int i = kernelTest.mIterations; i > 0; --i) {
                    for (auto &oneTest : kernelTest.mInvocationTests) {
                        InvocationTest::test_fn testFn = (result.second.mCompiledCorrectly ? oneTest.mTestFn : failTestFn);
                        result.second.mInvocationResults.push_back(
                                InvocationTest::result(&oneTest,
                                                       testFn(kernel,
                                                              kernelTest.mArguments,
                                                              kernelTest.mIsVerbose)));
                    }
                }
            }
            catch (...) {
                result.second.mExceptionString = current_exception_to_string();
            }
        }

        return result;
    }

    ModuleTest::result test_module(clspv_utils::device& inDevice,
                                   const ModuleTest&    moduleTest) {
        ModuleTest::result result;
        result.first = &moduleTest;

        try {
            file_utils::AndroidAssetStream spvmapStream(moduleTest.mName + ".spvmap");
            if (!spvmapStream.good())
            {
                throw std::runtime_error("cannot open spvmap for " + moduleTest.mName);
            }

            // spvmap files may have been generated on a system which uses different line ending
            // conventions than the system on which the consumer runs. Safer to fetch lines
            // using a function which recognizes multiple line endings.
            crlf_savvy::crlf_filter_buffer filter(spvmapStream.rdbuf());
            spvmapStream.rdbuf(&filter);

            clspv_utils::module_spec_t moduleInterface = clspv_utils::createModuleSpec(spvmapStream);
            spvmapStream.close();

            file_utils::AndroidAssetStream spvStream(moduleTest.mName + ".spv");
            if (!spvStream.good())
            {
                throw std::runtime_error("cannot open spv for " + moduleTest.mName);
            }

            clspv_utils::kernel_module module(moduleTest.mName,
                                              spvStream,
                                              inDevice,
                                              moduleInterface);
            result.second.mLoadedCorrectly = true;
            spvStream.close();

            auto entryPoints = module.getEntryPoints();
            for (const auto& ep : entryPoints) {
                std::vector<const KernelTest*> entryTests;
                for (auto& kt : moduleTest.mKernelTests) {
                    if (kt.mEntryName == ep) {
                        entryTests.push_back(&kt);
                    }
                }

                if (entryTests.empty()) {
                    result.second.mUntestedEntryPoints.push_back(ep);
                }

                // Iterate through all entries for the entry point in the test map.
                for (auto epTest : entryTests) {
                    if (vk::Extent3D(0, 0, 0) == epTest->mWorkgroupSize) {
                        // vk::Extent3D(0, 0, 0) is a sentinel to skip this kernel entirely

                        KernelTest::result kernelResult;
                        kernelResult.first = epTest;
                        kernelResult.second.mSkipped = true;

                        result.second.mKernelResults.push_back(kernelResult);
                    } else {
                        result.second.mKernelResults.push_back(test_kernel(module, *epTest));
                    }
                }
            }
        }
        catch (...) {
            result.second.mExceptionString = current_exception_to_string();
        }

        return result;
    }

    InvocationTest createNullInvocationTest() {
        InvocationTest result;
        result.mVariation = "compile-only";
        result.mTestFn = null_invocation_test;
        return result;
    }

} // namespace test_utils
