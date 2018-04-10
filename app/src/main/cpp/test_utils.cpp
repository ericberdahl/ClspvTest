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
                                 InvocationResultSet&               resultSet) {
        for (; first != last; ++first) {
            (*first)(module, kernel, info, samplers, args, resultSet);
        }
    }

    KernelResult test_kernel(const clspv_utils::kernel_module&          module,
                             const std::string&                         entryPoint,
                             test_kernel_fn                             testFn,
                             const clspv_utils::WorkgroupDimensions&    numWorkgroups,
                             const sample_info&                         info,
                             const std::vector<std::string>&            args,
                             vk::ArrayProxy<const vk::Sampler>          samplers) {
        KernelResult kernelResult;
        kernelResult.mEntryName = entryPoint;
        kernelResult.mSkipped = false;

		try {
	        clspv_utils::kernel kernel(module, entryPoint, numWorkgroups);
	        kernelResult.mCompiledCorrectly = true;

			if (testFn) {
				testFn(module, kernel, info, samplers, args, kernelResult.mInvocations);
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

                const auto epTest = std::find_if(kernelTests.begin(), kernelTests.end(),
                                                 [&ep](const kernel_test_map &ktm) {
                                                     return ktm.entry == ep;
                                                 });

                clspv_utils::WorkgroupDimensions num_workgroups;
                if (epTest != kernelTests.end()) {
                    num_workgroups = epTest->workgroupSize;
                }

                if (0 == num_workgroups.x && 0 == num_workgroups.y) {
                    // WorkgroupDimensions(0, 0) is a sentinel to skip this kernel entirely

                    KernelResult kernelResult;
                    kernelResult.mEntryName = ep;
                    moduleResult.mKernels.push_back(std::move(kernelResult));
                } else {
                    moduleResult.mKernels.push_back(test_kernel(module,
                                                                ep,
                                                                epTest == kernelTests.end() ? nullptr : epTest->test,
                                                                num_workgroups,
                                                                info,
                                                                epTest->args,
                                                                samplers));
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
