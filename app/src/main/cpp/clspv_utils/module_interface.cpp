//
// Created by Eric Berdahl on 10/22/17.
//

#include "interface.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "opencl_types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <istream>
#include <limits>
#include <memory>

#include <sstream> // std::istringstream

namespace {
    using namespace clspv_utils;

    typedef std::pair<string, string> key_value_t;

    const auto kSpvMapArgType_ArgKind_Map = {
            std::make_pair("pod", arg_spec_t::kind_pod),
            std::make_pair("pod_ubo", arg_spec_t::kind_pod_ubo),
            std::make_pair("buffer", arg_spec_t::kind_buffer),
            std::make_pair("ro_image", arg_spec_t::kind_ro_image),
            std::make_pair("wo_image", arg_spec_t::kind_wo_image),
            std::make_pair("sampler", arg_spec_t::kind_sampler),
            std::make_pair("local", arg_spec_t::kind_local)
    };

    arg_spec_t::kind_t find_arg_kind(const string &argType) {
        auto found = std::find_if(std::begin(kSpvMapArgType_ArgKind_Map),
                                  std::end(kSpvMapArgType_ArgKind_Map),
                                  [&argType](decltype(kSpvMapArgType_ArgKind_Map)::const_reference p) {
                                      return argType == p.first;
                                  });
        if (found == std::end(kSpvMapArgType_ArgKind_Map)) {
            fail_runtime_error("unknown argType encountered");
        }
        return found->second;
    }

    string read_csv_field(std::istream& in) {
        string result;

        if (in.good()) {
            const bool is_quoted = (in.peek() == '"');

            if (is_quoted) {
                in.ignore(std::numeric_limits<std::streamsize>::max(), '"');
            }

            std::getline(in, result, is_quoted ? '"' : ',');

            if (is_quoted) {
                in.ignore(std::numeric_limits<std::streamsize>::max(), ',');
            }
        }

        return result;
    }

    key_value_t read_key_value_pair(std::istream& in) {
        return std::make_pair(read_csv_field(in), read_csv_field(in));
    };

    sampler_spec_t parse_spvmap_sampler(key_value_t tag, std::istream& in) {
        sampler_spec_t result;

        result.opencl_flags = std::atoi(tag.second.c_str());

        while (!in.eof()) {
            tag = read_key_value_pair(in);

            if ("descriptorSet" == tag.first) {
                result.descriptor_set = std::atoi(tag.second.c_str());
            } else if ("binding" == tag.first) {
                result.binding = std::atoi(tag.second.c_str());
            }
        }

        return result;
    }

    arg_spec_t parse_spvmap_kernel_arg(key_value_t tag, std::istream& in) {
        arg_spec_t result;

        while (!in.eof()) {
            tag = read_key_value_pair(in);

            if ("argOrdinal" == tag.first) {
                result.ordinal = std::atoi(tag.second.c_str());
            } else if ("descriptorSet" == tag.first) {
                result.descriptor_set = std::atoi(tag.second.c_str());
            } else if ("binding" == tag.first) {
                result.binding = std::atoi(tag.second.c_str());
            } else if ("offset" == tag.first) {
                result.offset = std::atoi(tag.second.c_str());
            } else if ("argKind" == tag.first) {
                result.kind = find_arg_kind(tag.second);
            } else if ("arrayElemSize" == tag.first) {
                // arrayElemSize is ignored by clspvtest
            } else if ("arrayNumElemSpecId" == tag.first) {
                result.spec_constant = std::atoi(tag.second.c_str());
            }

        }

        return result;
    }

} // anonymous namespace

namespace clspv_utils {

    const kernel_spec_t* findKernelSpec(const string&                       name,
                                        const module_spec_t::kernel_list&   kernels)
    {
        auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                   [&name](const kernel_spec_t &iter) {
                                       return iter.mName == name;
                                   });

        return (kernel == kernels.end() ? nullptr : &(*kernel));
    }

    kernel_spec_t* findKernelSpec(const string&                 name,
                                  module_spec_t::kernel_list&   kernels)
    {
        return const_cast<kernel_spec_t*>(findKernelSpec(name, const_cast<const module_spec_t::kernel_list&>(kernels)));
    }

    module_spec_t createModuleSpec(std::istream& in)
    {
        module_spec_t result;

        /*
         * TODO Change file reading.
         * Parse each line into vector of key-value pairs.
         */

        kernel_spec_t* recentKernel = nullptr;

        string line;
        while (!in.eof()) {
            std::getline(in, line);

            std::istringstream in_line(line);
            auto tag = read_key_value_pair(in_line);
            if ("sampler" == tag.first) {
                result.mSamplers.push_back(parse_spvmap_sampler(tag, in_line));
            } else if ("kernel" == tag.first) {
                if (!recentKernel || recentKernel->mName != tag.second)
                {
                    recentKernel = findKernelSpec(tag.second, result.mKernels);
                    if (!recentKernel) {
                        result.mKernels.push_back(kernel_spec_t{ tag.second, kernel_spec_t::arg_list() });
                        recentKernel = &result.mKernels.back();
                    }
                }
                assert(recentKernel);

                recentKernel->mArguments.push_back(parse_spvmap_kernel_arg(tag, in_line));
            }
        }

        // Ensure that the literal samplers are sorted by increasing binding number. This will be
        // important if the sequence is later used to determine whether a cached sampler descriptor
        // set can be re-used for this module.
        std::sort(result.mSamplers.begin(), result.mSamplers.end(), [](const sampler_spec_t& lhs, const sampler_spec_t& rhs) {
            return lhs.binding < rhs.binding;
        });

        for (auto& k : result.mKernels) {
            standardizeKernelArgumentOrder(k.mArguments);
        }

        validateModule(result);

        return result;
    }

    void validateModule(const module_spec_t& spec)
    {
        const int sampler_ds = getSamplersDescriptorSet(spec.mSamplers);
        for (auto& ls : spec.mSamplers) {
            // All literal samplers for a module need to be in the same descriptor set
            validateSampler(ls, sampler_ds);
        }

        // If there are literal samplers, the kernel arguments are in descriptor set 1, otherwise
        // they are in descriptor set 0
        const int kernel_ds = (sampler_ds > 0 ? 1 : 0);
        for (auto& k : spec.mKernels) {
            validateKernel(k, kernel_ds);
        }

    }

    int getSamplersDescriptorSet(const module_spec_t::sampler_list& samplers) {
        auto found = std::find_if(samplers.begin(), samplers.end(),
                                  [](const sampler_spec_t &ss) {
                                      return (-1 != ss.descriptor_set);
                                  });
        return (found == samplers.end() ? -1 : found->descriptor_set);
    }

    vector<string> getEntryPointNames(const module_spec_t::kernel_list& specs)
    {
        vector<string> result;

        std::transform(specs.begin(), specs.end(),
                       std::back_inserter(result),
                       [](const kernel_spec_t& k) { return k.mName; });

        return result;
    }

} // namespace clspv_utils
