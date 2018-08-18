//
// Created by Eric Berdahl on 4/26/18.
//

#include "test_manifest.hpp"

#include "copybuffertoimage_kernel.hpp"
#include "copyimagetobuffer_kernel.hpp"
#include "fillarraystruct_kernel.hpp"
#include "fill_kernel.hpp"
#include "readconstantdata_kernel.hpp"
#include "readlocalsize_kernel.hpp"
#include "resample2dimage_kernel.hpp"
#include "resample3dimage_kernel.hpp"
#include "strangeshuffle_kernel.hpp"
#include "testgreaterthanorequalto_kernel.hpp"

#include "util.hpp" // for LOGxx macros

namespace
{
    using namespace test_manifest;

    typedef test_utils::KernelTest::invocation_tests (series_gen_signature)();

    typedef std::function<series_gen_signature>  series_gen_fn;

    series_gen_fn createGenerator(std::function<test_utils::InvocationTest ()> delegate)
    {
        return series_gen_fn([delegate]() {
            return test_utils::KernelTest::invocation_tests({ delegate() });
        });
    }

    series_gen_fn createGenerator(series_gen_fn genFn)
    {
        return genFn;
    }

    test_utils::KernelTest::invocation_tests lookup_test_series(const std::string& testName)
    {
        static const auto test_map = {
                std::make_pair("copyBufferToImage",  createGenerator(copybuffertoimage_kernel::getAllTestVariants)),
                std::make_pair("copyImageToBuffer",  createGenerator(copyimagetobuffer_kernel::getAllTestVariants)),
                std::make_pair("fillarraystruct",    createGenerator(fillarraystruct_kernel::getAllTestVariants)),
                std::make_pair("fill",               createGenerator(fill_kernel::getAllTestVariants)),
                std::make_pair("fill<float4>",       createGenerator(fill_kernel::getTestVariant<gpu_types::float4>)),
                std::make_pair("fill<half4>",        createGenerator(fill_kernel::getTestVariant<gpu_types::half4>)),
                std::make_pair("resample2dimage",    createGenerator(resample2dimage_kernel::getAllTestVariants)),
                std::make_pair("resample3dimage",    createGenerator(resample3dimage_kernel::getAllTestVariants)),
                std::make_pair("readLocalSize",      createGenerator(readlocalsize_kernel::getAllTestVariants)),
                std::make_pair("readConstantData",   createGenerator(readconstantdata_kernel::getAllTestVariants)),
                std::make_pair("strangeShuffle",     createGenerator(strangeshuffle_kernel::getAllTestVariants)),
                std::make_pair("testGtEq",           createGenerator(testgreaterthanorequalto_kernel::getAllTestVariants)),
        };

        test_utils::KernelTest::invocation_tests result;

        auto found = std::find_if(std::begin(test_map), std::end(test_map),
                                  [&testName](decltype(test_map)::const_reference entry){
                                      return testName == entry.first;
                                  });
        if (found != std::end(test_map))
        {
            result = found->second();
        }

        return result;
    }

    void read_module_op(std::istream& is, manifest_t& manifest)
    {
        // add module to list of modules to load
        test_utils::ModuleTest moduleEntry;
        is >> moduleEntry.mName;

        manifest.tests.push_back(moduleEntry);
    }

    void read_skip_op(std::istream& is, manifest_t& manifest)
    {
        if (manifest.tests.empty())
        {
            throw std::runtime_error("no module for skip");
        }

        // skip kernel in module
        test_utils::KernelTest skipEntry;
        skipEntry.mWorkgroupSize = vk::Extent3D(0, 0, 0);

        is >> skipEntry.mEntryName;

        manifest.tests.back().mKernelTests.push_back(skipEntry);
    }

    void read_vkvalidation_op(std::istream& is, manifest_t& manifest)
    {
        // turn vulkan validation layers on/off
        std::string on_off;
        is >> on_off;

        if (on_off == "all")
        {
            manifest.use_validation_layer = true;
        }
        else if (on_off == "none")
        {
            manifest.use_validation_layer = false;
        }
        else
        {
            throw std::runtime_error("unrecognized vkValidation value");
        }
    }

    bool read_verbosity_op(std::istream& is)
    {
        bool result = false;

        // set verbosity of tests
        std::string verbose_level;
        is >> verbose_level;

        if (verbose_level == "full")
        {
            result = true;
        }
        else if (verbose_level == "silent")
        {
            result = false;
        }
        else
        {
            throw std::runtime_error("unrecognized verbosity value");
        }

        return result;
    }

    unsigned int read_itereations_op(std::istream& is)
    {
        // set number of iterations for tests
        int iterations_requested;
        is >> iterations_requested;

        if (0 >= iterations_requested)
        {
            throw std::runtime_error("illegal iteration count requested");
        }

        return iterations_requested;
    }

    void read_test_op(std::istream&         is,
                      const std::string&    op,
                      manifest_t&           manifest,
                      bool                  verbose,
                      unsigned int          iterations)
    {
        if (manifest.tests.empty())
        {
            throw std::runtime_error("no module for test");
        }

        test_utils::KernelTest testEntry;
        testEntry.mIsVerbose = verbose;
        testEntry.mIterations = iterations;

        std::string testName;
        is >> testEntry.mEntryName
           >> testName
           >> testEntry.mWorkgroupSize.width
           >> testEntry.mWorkgroupSize.height;
        if (op == "test3d")
        {
            is >> testEntry.mWorkgroupSize.depth;
        }
        else
        {
            testEntry.mWorkgroupSize.depth = 1;
        }

        while (!is.eof())
        {
            std::string arg;
            is >> arg;

            // comment delimiter halts collection of test arguments
            if (arg[0] == '#') break;

            testEntry.mArguments.push_back(arg);
        }

        testEntry.mInvocationTests = lookup_test_series(testName);

        if (testEntry.mInvocationTests.empty())
        {
            throw std::runtime_error("cannot find tests " + testName);
        }
        if (1 > testEntry.mWorkgroupSize.width || 1 > testEntry.mWorkgroupSize.height || 1 > testEntry.mWorkgroupSize.depth)
        {
            std::ostringstream os;
            os << "bad workgroup dimensions {"
                  << testEntry.mWorkgroupSize.width
               << ',' << testEntry.mWorkgroupSize.height
               << ',' << testEntry.mWorkgroupSize.depth
               << '}';

            throw std::runtime_error(os.str());
        }

        manifest.tests.back().mKernelTests.push_back(testEntry);
    }

    void ensure_all_entries_tested(test_utils::ModuleTest& moduleTest)
    {
        clspv_utils::kernel_module module(moduleTest.mName);

        for (auto& entryPoint : module.getEntryPoints())
        {
            auto found = std::find_if(moduleTest.mKernelTests.begin(), moduleTest.mKernelTests.end(),
                         [&entryPoint](const test_utils::KernelTest& kt) {
                return kt.mEntryName == entryPoint;
            });

            if (found == moduleTest.mKernelTests.end())
            {
                test_utils::KernelTest loadOnlyTest;
                loadOnlyTest.mEntryName = entryPoint;
                loadOnlyTest.mWorkgroupSize = vk::Extent3D(1, 1, 1);
                loadOnlyTest.mInvocationTests.push_back(test_utils::createNullInvocationTest());

                moduleTest.mKernelTests.push_back(loadOnlyTest);
            }
        }
    }
}

namespace test_manifest
{

    test_manifest::results run(const manifest_t &manifest,
                               clspv_utils::device_t &device)
    {
        test_manifest::results results;

        for (auto& m : manifest.tests)
        {
            results.push_back(test_utils::test_module(device, m));
        }

        return results;
    }

    manifest_t read(const std::string &inManifest)
    {
        std::istringstream is(inManifest);
        return read(is);
    }

    manifest_t read(std::istream &in)
    {
        manifest_t result;
        unsigned int iterations = 1;
        bool verbose = false;

        while (!in.eof())
        {
            std::string line;
            std::getline(in, line);

            try
            {
                std::istringstream in_line(line);

                std::string op;
                in_line >> op;
                if (op.empty() || op[0] == '#')
                {
                    // line is either blank or a comment, skip it
                }
                else if (op == "module")
                {
                    read_module_op(in_line, result);
                }
                else if (op == "test" || op == "test2d" || op == "test3d")
                {
                    read_test_op(in_line, op, result, verbose, iterations);
                }
                else if (op == "skip")
                {
                    read_skip_op(in_line, result);
                }
                else if (op == "vkValidation")
                {
                    read_vkvalidation_op(in_line, result);
                }
                else if (op == "verbosity")
                {
                    verbose = read_verbosity_op(in_line);
                }
                else if (op == "iterations")
                {
                    iterations = read_itereations_op(in_line);
                }
                else if (op == "end")
                {
                    // terminate reading the manifest
                    break;
                }
                else
                {
                    throw std::runtime_error("ill-formed line");
                }
            }
            catch (const std::exception& e)
            {
                LOGE("%s: Error '%s' from command '%s'",
                     __func__,
                     e.what(),
                     line.c_str());
            }

        }

        for (auto& mt : result.tests)
        {
            ensure_all_entries_tested(mt);
        }

        return result;
    }
}
