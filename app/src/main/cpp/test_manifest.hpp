//
// Created by Eric Berdahl on 4/26/18.
//

#ifndef CLSPVTEST_TEST_MANIFEST_HPP
#define CLSPVTEST_TEST_MANIFEST_HPP

#include "clspv_utils.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace test_manifest {

    struct manifest_t {
        bool                                        use_validation_layer = true;
        std::vector<test_utils::module_test_bundle> tests;
    };

    manifest_t read(const std::string& inManifest);

    manifest_t read(std::istream& in);

    void run(const manifest_t&              manifest,
             clspv_utils::device_t&         info,
             test_utils::ModuleResultSet&   resultSet);
}

#endif //CLSPVTEST_TEST_MANIFEST_HPP
