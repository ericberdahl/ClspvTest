//
// Created by Eric Berdahl on 10/31/17.
//

#include "test_utils.hpp"

namespace {
    std::string current_exception_to_string()
    {
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
}

namespace test_utils {

    KernelResult test_kernel(clspv_utils::kernel_module&    module,
                             const kernel_test_map&         kernelTest) {
        KernelResult kernelResult;
        kernelResult.mEntryName = kernelTest.entry;
        kernelResult.mSkipped = false;

		try {
	        clspv_utils::kernel kernel(module, kernelTest.entry, kernelTest.workgroupSize);
	        kernelResult.mCompiledCorrectly = true;

			if (!kernelTest.tests.empty()) {
                kernelResult.mIterations = kernelTest.iterations;
                for (unsigned int i = kernelTest.iterations; i > 0; --i) {
                    for (auto oneTest : kernelTest.tests) {
                        kernelResult.mInvocations.push_back(oneTest(kernel,
                                                                    kernelTest.args,
                                                                    kernelTest.verbose));
                    }
                }
			}
		}
        catch (...) {
            kernelResult.mExceptionString = current_exception_to_string();
        }

        return kernelResult;
    }

    ModuleResult test_module(clspv_utils::device_t&                 device,
                             const std::string&                     moduleName,
                             const std::vector<kernel_test_map>&    kernelTests) {
        ModuleResult moduleResult;
        moduleResult.mModuleName = moduleName;

        try {
            clspv_utils::kernel_module module(device, moduleName);
            moduleResult.mLoadedCorrectly = true;

            std::vector<std::string> entryPoints(module.getEntryPoints());
            for (auto ep : entryPoints) {
                std::vector<kernel_test_map> entryTests;
                std::copy_if(kernelTests.begin(), kernelTests.end(),
                             std::back_inserter(entryTests),
                             [&ep](const kernel_test_map &ktm) {
                                 return ktm.entry == ep;
                             });

                if (entryTests.empty()) {
                    // This entry point was not called out specifically in the test map. Just
                    // compile the kernel with a default workgroup, but don't execute any specific
                    // test.
                    kernel_test_map dummyMap;
                    dummyMap.entry = ep;

                    entryTests.push_back(dummyMap);
                }

                // Iterate through all entries for the entry point in the test map.
                for (auto& epTest : entryTests) {
                    if (vk::Extent3D(0, 0, 0) == epTest.workgroupSize) {
                        // vk::Extent3D(0, 0, 0) is a sentinel to skip this kernel entirely

                        KernelResult kernelResult;
                        kernelResult.mEntryName = ep;

                        moduleResult.mKernels.push_back(std::move(kernelResult));
                    } else {
                        KernelResult kernelResult = test_kernel(module, epTest);

                        moduleResult.mKernels.push_back(kernelResult);
                    }
                }
            }
        }
        catch (...) {
            moduleResult.mExceptionString = current_exception_to_string();
        }

        return moduleResult;
    }

} // namespace test_utils
