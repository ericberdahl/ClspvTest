//
// Created by Eric Berdahl on 4/26/18.
//

#ifndef CLSPVTEST_TEST_MANIFEST_HPP
#define CLSPVTEST_TEST_MANIFEST_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace test_manifest {

    struct manifest_t {
        bool                                use_validation_layer = true;
        std::vector<test_utils::ModuleTest> tests;
    };

    typedef std::vector<test_utils::ModuleTest::result> results;

    manifest_t read(const std::string& inManifest);

    manifest_t read(std::istream& in);

    results run(const manifest_t&       manifest,
                clspv_utils::device&    info);
}

#endif //CLSPVTEST_TEST_MANIFEST_HPP
