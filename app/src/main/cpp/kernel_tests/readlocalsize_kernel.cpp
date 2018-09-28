//
// Created by Eric Berdahl on 10/31/17.
//

#include "readlocalsize_kernel.hpp"

#include "clspv_utils/kernel.hpp"

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

    std::vector<std::int32_t> compute_expected_group_id_z(int width, int height, int pitch, int group_depth) {
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

    std::vector<std::int32_t> compute_expected_local_id_z(int width, int height, int pitch, int group_depth) {
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

    std::vector<std::int32_t> compute_expected_local_size_z(int width, int height, int pitch, int group_depth) {
        std::vector<std::int32_t> expected(pitch * height);

        auto rowIter = expected.begin();
        for (int i = 0; i < height; ++i) {
            std::fill_n(rowIter, width, 1);
            rowIter = std::next(rowIter, pitch);
        }

        return expected;
    }

    std::vector<std::int32_t> compute_expected_results(idtype_t             idtype,
                                                       int                  buffer_width,
                                                       int                  buffer_height,
                                                       const vk::Extent3D&  workgroupSize) {
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
                expectedResults = compute_expected_local_size_x(buffer_width, buffer_height, buffer_width, workgroupSize.width);
                break;

            case idtype_localsize_y:
                expectedResults = compute_expected_local_size_y(buffer_width, buffer_height, buffer_width, workgroupSize.height);
                break;

            case idtype_localsize_z:
                expectedResults = compute_expected_local_size_z(buffer_width, buffer_height, buffer_width, workgroupSize.depth);
                break;

            case idtype_groupid_x:
                expectedResults = compute_expected_group_id_x(buffer_width, buffer_height, buffer_width, workgroupSize.width);
                break;

            case idtype_groupid_y:
                expectedResults = compute_expected_group_id_y(buffer_width, buffer_height, buffer_width, workgroupSize.height);
                break;

            case idtype_groupid_z:
                expectedResults = compute_expected_group_id_z(buffer_width, buffer_height, buffer_width, workgroupSize.depth);
                break;

            case idtype_localid_x:
                expectedResults = compute_expected_local_id_x(buffer_width, buffer_height, buffer_width, workgroupSize.width);
                break;

            case idtype_localid_y:
                expectedResults = compute_expected_local_id_y(buffer_width, buffer_height, buffer_width, workgroupSize.height);
                break;

            case idtype_localid_z:
                expectedResults = compute_expected_local_id_z(buffer_width, buffer_height, buffer_width, workgroupSize.depth);
                break;

            default:
                throw std::runtime_error("unknown idtype requested");
        }

        return expectedResults;
    }
}

namespace readlocalsize_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    outLocalSizes,
           int                              inWidth,
           int                              inHeight,
           int                              inPitch,
           idtype_t                         inIdType) {
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

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().getDevice(),
                                                  kernel.getDevice().getMemoryProperties(),
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->width = inWidth;
        scalars->height = inHeight;
        scalars->pitch = inPitch;
        scalars->idtype = inIdType;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (inWidth + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (inHeight + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(outLocalSizes);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose)
    {
        test_utils::InvocationResult    invocationResult;
        auto& device = kernel.getDevice();

        invocationResult.mParameters = (args.empty() ? std::string("global_id_x") : args[0]);

        const idtype_t idtype = idtype_from_string(invocationResult.mParameters);
        const vk::Extent3D bufferExtent(64, 64, 1);

        // allocate data buffer
        auto num_elements = bufferExtent.width * bufferExtent.height * bufferExtent.depth;
        const std::size_t buffer_size = num_elements * sizeof(std::int32_t);
        vulkan_utils::storage_buffer dst_buffer(device.getDevice(), device.getMemoryProperties(), buffer_size);

        auto dstBufferMap = dst_buffer.map<std::int32_t>();
        test_utils::fill_random_pixels<std::int32_t>(dstBufferMap.get(), dstBufferMap.get() + num_elements);
        dstBufferMap.reset();

        auto expectedResults = compute_expected_results(idtype,
                                                        bufferExtent.width,
                                                        bufferExtent.height,
                                                        kernel.getWorkgroupSize());

        invocationResult.mExecutionTime = invoke(kernel,
                                                 dst_buffer,
                                                 bufferExtent.width,
                                                 bufferExtent.height,
                                                 bufferExtent.width,
                                                 idtype);

        dstBufferMap = dst_buffer.map<std::int32_t>();
        test_utils::check_results(expectedResults.data(),
                                  dstBufferMap.get(),
                                  bufferExtent,
                                  bufferExtent.width,
                                  verbose,
                                  invocationResult);

        return invocationResult;
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        test_utils::InvocationTest t({ "", test });
        return test_utils::KernelTest::invocation_tests({ t });
    }

}
