//
// Created by Eric Berdahl on 10/31/17.
//

#include "readlocalsize_kernel.hpp"

namespace {
    using namespace readlocalsize_kernel;

    auto idtype_string_map = {
            std::make_pair("global_size_x", idtype_globalsize_x),
            std::make_pair("global_size_y", idtype_globalsize_y),
            std::make_pair("global_size_z", idtype_globalsize_z),

            std::make_pair("local_size_x", idtype_localsize_x),
            std::make_pair("local_size_y", idtype_localsize_y),
            std::make_pair("local_size_z", idtype_localsize_z),

            std::make_pair("global_id_x", idtype_globalid_x),
            std::make_pair("global_id_y", idtype_globalid_y),
            std::make_pair("global_id_z", idtype_globalid_z),

            std::make_pair("group_id_x", idtype_groupid_x),
            std::make_pair("group_id_y", idtype_groupid_y),
            std::make_pair("group_id_z", idtype_groupid_z),

            std::make_pair("local_id_x", idtype_localid_x),
            std::make_pair("local_id_y", idtype_localid_y),
            std::make_pair("local_id_z", idtype_localid_z)
    };

    idtype_t idtype_from_string(const std::string& s) {
        auto found = std::find_if(idtype_string_map.begin(), idtype_string_map.end(),
                                  [&s](decltype(idtype_string_map)::const_reference item) {
                                      return item.first == s;
                                  });
        if (found == idtype_string_map.end()) {
            found = idtype_string_map.begin();
        }
        return found->second;
    }

    std::vector<std::int32_t> compute_expected_global_id_x(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::iota(rowIter, std::next(rowIter, width), 0);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_global_id_y(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, i);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_global_id_z(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, 0);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_global_size_x(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, width);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_global_size_y(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, height);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_global_size_z(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, 1);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_group_id_x(int width, int height, int pitch, int group_width) {
        std::vector<std::int32_t> expected(pitch * height);

        const int num_groups = (width + group_width - 1) / group_width;
        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            for (int g = 0; g < num_groups; ++g) {
                std::fill_n(std::next(rowIter, g*group_width), group_width, g);
            }

            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_group_id_y(int width, int height, int pitch, int group_height) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, i / group_height);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_group_id_z(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, 0);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_local_id_x(int width, int height, int pitch, int group_width) {
        std::vector<std::int32_t> expected(pitch * height);

        const int num_groups = (width + group_width - 1) / group_width;
        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            for (int g = 0; g < num_groups; ++g) {
                auto start = std::next(rowIter, g*group_width);
                auto end = std::next(start, group_width);
                std::iota(start, end, 0);
            }

            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_local_id_y(int width, int height, int pitch, int group_height) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, i % group_height);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_local_id_z(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, 0);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_local_size_x(int width, int height, int pitch, int group_width) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, group_width);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_local_size_y(int width, int height, int pitch, int group_height) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, group_height);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_local_size_z(int width, int height, int pitch) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, 1);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_results(idtype_t                                 idtype,
                                                       int                                      buffer_width,
                                                       int                                      buffer_height,
                                                       const clspv_utils::WorkgroupDimensions&  workgroupSize) {
        std::vector<std::int32_t> expectedResults;
        switch (idtype) {
            case idtype_globalid_x:
                expectedResults = compute_expected_global_id_x(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_globalid_y:
                expectedResults = compute_expected_global_id_y(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_globalid_z:
                expectedResults = compute_expected_global_id_z(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_globalsize_x:
                expectedResults = compute_expected_global_size_x(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_globalsize_y:
                expectedResults = compute_expected_global_size_y(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_globalsize_z:
                expectedResults = compute_expected_global_size_z(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_localsize_x:
                expectedResults = compute_expected_local_size_x(buffer_width, buffer_height, buffer_width, workgroupSize.x);
                break;

            case idtype_localsize_y:
                expectedResults = compute_expected_local_size_y(buffer_width, buffer_height, buffer_width, workgroupSize.y);
                break;

            case idtype_localsize_z:
                expectedResults = compute_expected_local_size_z(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_groupid_x:
                expectedResults = compute_expected_group_id_x(buffer_width, buffer_height, buffer_width, workgroupSize.x);
                break;

            case idtype_groupid_y:
                expectedResults = compute_expected_group_id_y(buffer_width, buffer_height, buffer_width, workgroupSize.y);
                break;

            case idtype_groupid_z:
                expectedResults = compute_expected_group_id_z(buffer_width, buffer_height, buffer_width);
                break;

            case idtype_localid_x:
                expectedResults = compute_expected_local_id_x(buffer_width, buffer_height, buffer_width, workgroupSize.x);
                break;

            case idtype_localid_y:
                expectedResults = compute_expected_local_id_y(buffer_width, buffer_height, buffer_width, workgroupSize.y);
                break;

            case idtype_localid_z:
                expectedResults = compute_expected_local_id_z(buffer_width, buffer_height, buffer_width);
                break;

            default:
                throw std::runtime_error("unknown idtype requested");
        }

        return expectedResults;
    }
}

namespace readlocalsize_kernel {

    clspv_utils::execution_time_t
    invoke(const clspv_utils::kernel_module&    module,
           const clspv_utils::kernel&           kernel,
           const sample_info&                   info,
           vk::ArrayProxy<const vk::Sampler>    samplers,
           vk::Buffer                           outLocalSizes,
           int                                  inWidth,
           int                                  inHeight,
           int                                  inPitch,
           idtype_t                             inIdType) {
        struct scalar_args {
            int width;  // offset 0
            int height; // offset 4
            int pitch;  // offset 8
            int idtype; // offset 12
        };
        static_assert(0 == offsetof(scalar_args, width), "width offset incorrect");
        static_assert(4 == offsetof(scalar_args, height), "height offset incorrect");
        static_assert(8 == offsetof(scalar_args, pitch), "pitch offset incorrect");
        static_assert(12 == offsetof(scalar_args, idtype), "idtype offset incorrect");

        const scalar_args scalars = {
                inWidth,
                inHeight,
                inPitch,
                inIdType
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (scalars.width + workgroup_sizes.x - 1) / workgroup_sizes.x,
                (scalars.height + workgroup_sizes.y - 1) / workgroup_sizes.y);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                              info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addStorageBufferArgument(outLocalSizes);
        invocation.addPodArgument(scalars);

        return invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    void test(const clspv_utils::kernel_module&  module,
              const clspv_utils::kernel&         kernel,
              const sample_info&                 info,
              vk::ArrayProxy<const vk::Sampler>  samplers,
              const std::vector<std::string>&    args,
              bool                               verbose,
              test_utils::InvocationResultSet&   resultSet)
    {
        test_utils::InvocationResult    invocationResult;

        invocationResult.mVariation = (args.empty() ? std::string("global_id_x") : args[0]);

        const idtype_t idtype = idtype_from_string(invocationResult.mVariation);
        const int buffer_height   = 64;
        const int buffer_width    = 64;

        // allocate data buffer
        const std::size_t buffer_size = buffer_width * buffer_height * sizeof(std::int32_t);
        vulkan_utils::storage_buffer dst_buffer(info, buffer_size);

        auto expectedResults = compute_expected_results(idtype,
                                                        buffer_width,
                                                        buffer_height,
                                                        kernel.getWorkgroupSize());

        invocationResult.mExecutionTime = invoke(module,
                                                 kernel,
                                                 info,
                                                 samplers,
                                                 *dst_buffer.buf,
                                                 buffer_width,
                                                 buffer_height,
                                                 buffer_width,
                                                 idtype);

        test_utils::check_results<std::int32_t,std::int32_t>(expectedResults.data(),
                                                             dst_buffer.mem,
                                                             buffer_width,
                                                             buffer_height,
                                                             buffer_width,
                                                             verbose,
                                                             invocationResult);

        resultSet.push_back(std::move(invocationResult));
    };
}
