//
// Created by Eric Berdahl on 10/31/17.
//

#include "test_utils.hpp"

namespace test_utils {
    const Results Results::sTestSuccess(1, 0, 0, 0, 0);
    const Results Results::sTestFailure(0, 1, 0, 0, 0);
    const Results Results::sKernelLoadSuccess(0, 0, 1, 0, 0);
    const Results Results::sKernelLoadSkip(0, 0, 0, 1, 0);
    const Results Results::sKernelLoadFail(0, 0, 0, 0, 1);

    Results test_kernel_invocation(const clspv_utils::kernel_module&    module,
                                   const clspv_utils::kernel&           kernel,
                                   test_kernel_fn                       testFn,
                                   const sample_info&                   info,
                                   vk::ArrayProxy<const vk::Sampler>    samplers,
                                   const options&                       opts) {
        Results result;

        if (testFn) {
            const std::string label = module.getName() + "/" + kernel.getEntryPoint();
            result = runInExceptionContext(label,
                                           "invoking kernel",
                                           [&]() {
                                               return testFn(module, kernel, info, samplers, opts);
                                           });

            if ((result.mNumTestSuccess > 0 && opts.logCorrect) ||
                (result.mNumTestFail > 0 && opts.logIncorrect)) {
                LOGE("%s: Successes=%d Failures=%d",
                     label.c_str(),
                     result.mNumTestSuccess, result.mNumTestFail);
            }
        }

        return result;
    }

    Results test_kernel_invocation(const clspv_utils::kernel_module&    module,
                                   const clspv_utils::kernel&           kernel,
                                   const test_kernel_fn*                first,
                                   const test_kernel_fn*                last,
                                   const sample_info&                   info,
                                   vk::ArrayProxy<const vk::Sampler>    samplers,
                                   const options&                       opts) {
        Results result;

        for (; first != last; ++first) {
            result += test_kernel_invocation(module, kernel, *first, info, samplers, opts);
        }

        return result;
    }

    Results test_kernel(const clspv_utils::kernel_module&       module,
                        const std::string&                      entryPoint,
                        test_kernel_fn                          testFn,
                        const clspv_utils::WorkgroupDimensions& numWorkgroups,
                        const sample_info&                      info,
                        vk::ArrayProxy<const vk::Sampler>       samplers,
                        const options&                          opts) {
        return runInExceptionContext(module.getName() + "/" + entryPoint,
                                     "compiling kernel",
                                     [&]() {
                                         Results results;

                                         clspv_utils::kernel kernel(module,
                                                                    entryPoint,
                                                                    numWorkgroups);
                                         results += Results::sKernelLoadSuccess;

                                         results += test_kernel_invocation(module,
                                                                           kernel,
                                                                           testFn,
                                                                           info,
                                                                           samplers,
                                                                           opts);

                                         return results;
                                     },
                                     Results::sKernelLoadFail);
    }

    Results test_module(const std::string&                  moduleName,
                        const std::vector<kernel_test_map>& kernelTests,
                        const sample_info&                  info,
                        vk::ArrayProxy<const vk::Sampler>   samplers,
                        const options&                      opts) {
        return runInExceptionContext(moduleName, "loading module", [&]() {
            Results result;

            clspv_utils::kernel_module module((VkDevice) *info.device, (VkDescriptorPool) *info.desc_pool, moduleName);
            result += Results::sTestSuccess;

            std::vector<std::string> entryPoints(module.getEntryPoints());
            for (auto ep : entryPoints) {
                const auto epTest = std::find_if(kernelTests.begin(), kernelTests.end(),
                                                 [&ep](const kernel_test_map& ktm) {
                                                     return ktm.entry == ep;
                                                 });

                clspv_utils::WorkgroupDimensions num_workgroups;
                if (epTest != kernelTests.end()) {
                    num_workgroups = epTest->workgroupSize;
                }

                if (0 == num_workgroups.x && 0 == num_workgroups.y) {
                    // WorkgroupDimensions(0, 0) is a sentinel to skip this kernel entirely
                    LOGI("%s/%s: Skipping kernel", moduleName.c_str(), ep.c_str());
                    result += Results::sKernelLoadSkip;
                }
                else {
                    result += test_kernel(
                            module,
                            ep,
                            epTest == kernelTests.end() ? nullptr : epTest->test,
                            num_workgroups,
                            info,
                            samplers,
                            opts);
                }
            }

            LOGI("%s: %u/%d kernel successes",
                 moduleName.c_str(),
                 result.mNumKernelLoadSuccess,
                 (int)entryPoints.size());

            return result;
        });
    }

} // namespace test_utils
