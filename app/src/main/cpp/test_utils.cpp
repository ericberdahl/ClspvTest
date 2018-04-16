//
// Created by Eric Berdahl on 10/31/17.
//

#include "test_utils.hpp"

namespace test_utils {

    void test_kernel_invocations(const clspv_utils::kernel_module&  module,
                                 const clspv_utils::kernel&         kernel,
                                 const test_kernel_fn*              first,
                                 const test_kernel_fn*              last,
                                 const sample_info&                 info,
                                 vk::ArrayProxy<const vk::Sampler>  samplers,
                                 const std::vector<std::string>&    args,
                                 bool                               verbose,
                                 InvocationResultSet&               resultSet) {
        for (; first != last; ++first) {
            (*first)(module, kernel, info, samplers, args, verbose, resultSet);
        }
    }

    KernelResult test_kernel(const clspv_utils::kernel_module&          module,
                             const std::string&                         entryPoint,
                             test_kernel_fn                             testFn,
                             const clspv_utils::WorkgroupDimensions&    numWorkgroups,
                             const sample_info&                         info,
                             const std::vector<std::string>&            args,
                             bool                                       verbose,
                             vk::ArrayProxy<const vk::Sampler>          samplers) {
        KernelResult kernelResult;
        kernelResult.mEntryName = entryPoint;
        kernelResult.mSkipped = false;

		try {
	        clspv_utils::kernel kernel(module, entryPoint, numWorkgroups);
	        kernelResult.mCompiledCorrectly = true;

			if (testFn) {
				testFn(module, kernel, info, samplers, args, verbose, kernelResult.mInvocations);
			}
		}
        catch (const vk::SystemError &e) {
            std::ostringstream os;
            os << "vk::SystemError : " << e.code() << " (" << e.code().message() << ')';
            kernelResult.mExceptionString = os.str();
        }
        catch (const std::system_error &e) {
            std::ostringstream os;
            os << "std::system_error : " << e.code() << " (" << e.code().message() << ')';
            kernelResult.mExceptionString = os.str();
        }
        catch (const std::exception &e) {
            std::ostringstream os;
            os << "std::exception : " << e.what();
            kernelResult.mExceptionString = os.str();
        }
        catch (...) {
            kernelResult.mExceptionString = "unknonwn exception";
        }

        return kernelResult;
    }

    ModuleResult test_module(const std::string&             moduleName,
                     const std::vector<kernel_test_map>&    kernelTests,
                     const sample_info&                     info,
                     vk::ArrayProxy<const vk::Sampler>      samplers) {
        ModuleResult moduleResult;
        moduleResult.mModuleName = moduleName;

        try {
            clspv_utils::kernel_module module(*info.device, *info.desc_pool, moduleName);
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
                    clspv_utils::WorkgroupDimensions    num_workgroups;
                    std::vector<std::string>            emptyArgs;

                    moduleResult.mKernels.push_back(test_kernel(module,
                                                                ep,
                                                                nullptr,
                                                                num_workgroups,
                                                                info,
                                                                emptyArgs,
                                                                false,
                                                                samplers));
                }
                else {
                    // This entry point contained at least one test specified by the test map.
                    // Iterate through all entries for the entry point in the test map.

                    for (auto& epTest : entryTests) {
                        const clspv_utils::WorkgroupDimensions num_workgroups = epTest.workgroupSize;

                        if (0 == num_workgroups.x && 0 == num_workgroups.y) {
                            // WorkgroupDimensions(0, 0) is a sentinel to skip this kernel entirely

                            KernelResult kernelResult;
                            kernelResult.mEntryName = ep;

                            moduleResult.mKernels.push_back(std::move(kernelResult));
                        } else {
                            KernelResult kernelResult = test_kernel(module,
                                                                    ep,
                                                                    epTest.test,
                                                                    num_workgroups,
                                                                    info,
                                                                    epTest.args,
                                                                    epTest.verbose,
                                                                    samplers);

                            moduleResult.mKernels.push_back(kernelResult);
                        }
                    }
                }
            }
        }
        catch (const vk::SystemError &e) {
            std::ostringstream os;
            os << "vk::SystemError : " << e.code() << " (" << e.code().message() << ')';
            moduleResult.mExceptionString = os.str();
        }
        catch (const std::system_error &e) {
            std::ostringstream os;
            os << "std::system_error : " << e.code() << " (" << e.code().message() << ')';
            moduleResult.mExceptionString = os.str();
        }
        catch (const std::exception &e) {
            std::ostringstream os;
            os << "std::exception : " << e.what();
            moduleResult.mExceptionString = os.str();
        }
        catch (...) {
            moduleResult.mExceptionString = "unknonwn exception";
        }

        return moduleResult;
    }

} // namespace test_utils
