#ifndef CLSPVUTILS_INTERFACE_HPP
#define CLSPVUTILS_INTERFACE_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"

#include <vulkan/vulkan.hpp>

#include <iosfwd>

namespace clspv_utils {

    struct arg_spec_t {
        enum kind {
            kind_unknown,
            kind_pod,
            kind_pod_ubo,
            kind_buffer,
            kind_ro_image,
            kind_wo_image,
            kind_sampler,
            kind_local
        };

        kind    mKind           = kind_unknown;
        int     mOrdinal        = -1;
        int     mDescriptorSet  = -1;
        int     mBinding        = -1;
        int     mOffset         = -1;
        int     mSpecConstant   = -1;
    };

    struct sampler_spec_t {
        int mOpenclFlags    = 0;
        int mDescriptorSet  = -1;
        int mBinding        = -1;
    };

    struct kernel_spec_t {
        typedef vector<arg_spec_t>  arg_list;

        string      mName;
        arg_list    mArguments;
    };

    struct module_spec_t {
        typedef vector<sampler_spec_t>  sampler_list;
        typedef vector<kernel_spec_t>   kernel_list;

        sampler_list  mSamplers;
        kernel_list   mKernels;
    };

    /*
     * module_spect_t functions
     */

    module_spec_t           createModuleSpec(std::istream& spvmapStream);

    /*
     * module_spec_t::kernel_list functions
     */

    const kernel_spec_t*    findKernelSpec(const string&                        name,
                                           const module_spec_t::kernel_list&    kernels);

    kernel_spec_t*          findKernelSpec(const string&                name,
                                           module_spec_t::kernel_list&  kernels);

    vector<string>          getEntryPointNames(const module_spec_t::kernel_list& specs);

    /*
     * module_spec_t::sampler_list functions
     */

    int                     getSamplersDescriptorSet(const module_spec_t::sampler_list& spec);

    /*
     * kernel_spec_t::arg_list functions
     */

    vk::UniqueDescriptorSetLayout createKernelArgumentDescriptorLayout(const kernel_spec_t::arg_list&   arguments,
                                                                       vk::Device                       inDevice);

    /*
     * Sort the args such that pods are grouped together at the end of the sequence, and that
     * the non-pod and pod groups are each individually sorted by increasing ordinal
     */
    void    standardizeKernelArgumentOrder(kernel_spec_t::arg_list& arguments);

    int     getKernelArgumentDescriptorSet(const kernel_spec_t::arg_list& arguments);

    /*
     * arg_spec_t::kind_t functions
     */

    vk::DescriptorType  getDescriptorType(arg_spec_t::kind argKind);

    /*
     * OpenCL sampler flags functions
     */

    vk::SamplerAddressMode  getSamplerAddressMode(int opencl_flags);

    vk::Filter              getSamplerFilter(int opencl_flags);

    vk::Bool32              isSamplerUnnormalizedCoordinates(int opencl_flags);

    bool                    isSamplerSupported(int opencl_flags);

    vk::UniqueSampler       createCompatibleSampler(vk::Device device, int opencl_flags);

    /*
     * Validation functions
     */

    void    validateKernelArg(const arg_spec_t &arg);

    void    validateSampler(const sampler_spec_t& spec, int requiredDescriptorSet = -1);

    void    validateKernel(const kernel_spec_t& spec, int requiredDescriptorSet = -1);

    void    validateModule(const module_spec_t& spec);
}

#endif // CLSPVUTILS_INTERFACE_HPP
