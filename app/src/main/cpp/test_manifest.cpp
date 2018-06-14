//
// Created by Eric Berdahl on 4/26/18.
//

#include "test_manifest.hpp"

#include "copybuffertoimage_kernel.hpp"
#include "copyimagetobuffer_kernel.hpp"
#include "fill_kernel.hpp"
#include "readconstantdata_kernel.hpp"
#include "readlocalsize_kernel.hpp"
#include "strangeshuffle_kernel.hpp"
#include "testgreaterthanorequalto_kernel.hpp"

#include "util.hpp" // for LOGxx macros

namespace {
    const auto test_map = {
            std::make_pair("readLocalSize",      readlocalsize_kernel::test),
            std::make_pair("fill",               fill_kernel::test_series),
            std::make_pair("fill<float4>",       fill_kernel::test<gpu_types::float4>),
            std::make_pair("fill<half4>",        fill_kernel::test<gpu_types::half4>),
            std::make_pair("copyImageToBuffer",  copyimagetobuffer_kernel::test_matrix),
            std::make_pair("copyBufferToImage",  copybuffertoimage_kernel::test_matrix),
            std::make_pair("readConstantData",   readconstantdata_kernel::test_all),
            std::make_pair("testGtEq",           testgreaterthanorequalto_kernel::test_all),
            std::make_pair("strangeShuffle",     strangeshuffle_kernel::test),
    };

    test_utils::test_kernel_fn lookup_test_fn(const std::string& testName) {
        test_utils::test_kernel_fn result = nullptr;

        auto found = std::find_if(std::begin(test_map), std::end(test_map),
                                  [&testName](decltype(test_map)::const_reference entry){
                                      return testName == entry.first;
                                  });
        if (found != std::end(test_map)) {
            result = found->second;
        }

        return result;
    }
}

namespace test_manifest {
    void run(const manifest_t &manifest,
             clspv_utils::device_t &device,
             test_utils::ModuleResultSet &resultSet) {
        for (auto m : manifest.tests) {
            resultSet.push_back(test_utils::test_module(device, m.name, m.kernelTests));
        }
    }

    manifest_t read(const std::string &inManifest) {
        std::istringstream is(inManifest);
        return read(is);
    }

    manifest_t read(std::istream &in) {
        manifest_t result;
        unsigned int iterations = 1;
        bool verbose = false;

        test_utils::module_test_bundle *currentModule = NULL;
        while (!in.eof()) {
            std::string line;
            std::getline(in, line);

            std::istringstream in_line(line);

            std::string op;
            in_line >> op;
            if (op.empty() || op[0] == '#') {
                // line is either blank or a comment, skip it
            } else if (op == "module") {
                // add module to list of modules to load
                test_utils::module_test_bundle moduleEntry;
                in_line >> moduleEntry.name;

                result.tests.push_back(moduleEntry);
                currentModule = &result.tests.back();
            } else if (op == "test") {
                // test kernel in module
                if (currentModule) {
                    test_utils::kernel_test_map testEntry;
                    testEntry.verbose = verbose;
                    testEntry.iterations = iterations;

                    std::string testName;
                    in_line >> testEntry.entry
                            >> testName
                            >> testEntry.workgroupSize.width
                            >> testEntry.workgroupSize.height;

                    while (!in_line.eof()) {
                        std::string arg;
                        in_line >> arg;

                        // comment delimiter halts collection of test arguments
                        if (arg[0] == '#') break;

                        testEntry.args.push_back(arg);
                    }

                    testEntry.test = lookup_test_fn(testName);

                    bool lineIsGood = true;

                    if (!testEntry.test) {
                        LOGE("%s: cannot find test '%s' from command '%s'",
                             __func__,
                             testName.c_str(),
                             line.c_str());
                        lineIsGood = false;
                    }
                    if (1 > testEntry.workgroupSize.width || 1 > testEntry.workgroupSize.height) {
                        LOGE("%s: bad workgroup dimensions {%d,%d} from command '%s'",
                             __func__,
                             testEntry.workgroupSize.width,
                             testEntry.workgroupSize.height,
                             line.c_str());
                        lineIsGood = false;
                    }

                    if (lineIsGood) {
                        currentModule->kernelTests.push_back(testEntry);
                    }
                } else {
                    LOGE("%s: no module for test '%s'", __func__, line.c_str());
                }
            } else if (op == "skip") {
                // skip kernel in module
                if (currentModule) {
                    test_utils::kernel_test_map skipEntry;
                    skipEntry.workgroupSize = vk::Extent2D(0, 0);

                    in_line >> skipEntry.entry;

                    currentModule->kernelTests.push_back(skipEntry);
                } else {
                    LOGE("%s: no module for skip '%s'", __func__, line.c_str());
                }
            } else if (op == "vkValidation") {
                // turn vulkan validation layers on/off
                std::string on_off;
                in_line >> on_off;

                if (on_off == "all") {
                    result.use_validation_layer = true;
                } else if (on_off == "none") {
                    result.use_validation_layer = false;
                } else {
                    LOGE("%s: unrecognized vkValidation token '%s'", __func__, on_off.c_str());
                }
            } else if (op == "verbosity") {
                // set verbosity of tests
                std::string verbose_level;
                in_line >> verbose_level;

                if (verbose_level == "full") {
                    verbose = true;
                } else if (verbose_level == "silent") {
                    verbose = false;
                } else {
                    LOGE("%s: unrecognized verbose level '%s'", __func__, verbose_level.c_str());
                }
            } else if (op == "iterations") {
                // set number of iterations for tests
                int iterations_requested;
                in_line >> iterations_requested;

                if (0 >= iterations_requested) {
                    LOGE("%s: illegal iteration count requested '%d'", __func__,
                         iterations_requested);
                } else {
                    iterations = iterations_requested;
                }
            } else if (op == "end") {
                // terminate reading the manifest
                break;
            } else {
                LOGE("%s: ignoring ill-formed line '%s'", __func__, line.c_str());
            }
        }

        return result;
    }

}