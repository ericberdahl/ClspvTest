//
// Created by Eric Berdahl on 10/31/17.
//

#include "generic_kernel.hpp"

#include "clspv_utils/kernel.hpp"

namespace {
    std::vector<std::uint8_t> hexToBytes(std::string hexString) {
        std::vector<std::uint8_t> result;

        if (0 != (hexString.length() % 2)) {
            clspv_utils::fail_runtime_error("hex string must have even number of characters");
        }

        result.reserve(hexString.length() / 2);

        for (std::string::size_type i = 0; i < hexString.length(); i +=2) {
            const std::string   byteString = hexString.substr(i, 2);
            const int           byteInt = std::stoi(byteString, nullptr, 16);

            assert(0 <= byteInt);
            assert(byteInt <= std::numeric_limits<std::uint8_t>::max());

            result.push_back(static_cast<std::uint8_t>(byteInt));
        }

        return result;
    }
}

namespace generic_kernel {

    Test::Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args)
    {
        auto& device = kernel.getDevice();

        auto arg = args.begin();

        if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
        mNumWorkgroups.width = std::atoi(arg->c_str());
        arg = std::next(arg);

        if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
        mNumWorkgroups.height = std::atoi(arg->c_str());
        arg = std::next(arg);

        if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
        mNumWorkgroups.depth = std::atoi(arg->c_str());
        arg = std::next(arg);

        for (; arg != args.end(); arg = std::next(arg)) {
            if (*arg == "-pb") {
                // add an uninitialized storage buffer argument
                arg = std::next(arg);
                if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
                const std::size_t bufferSize = std::atoi(arg->c_str());

                mStorageBuffers.push_back(vulkan_utils::createStorageBuffer(device.getDevice(),
                                                                            device.getMemoryProperties(),
                                                                            bufferSize));
                mArgOrder.push_back(kind_storageBuffer);
            }
            else if (*arg == "-sb") {
                // add a storage buffer argument
                arg = std::next(arg);
                if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
                const auto bufferContents = hexToBytes(*arg);

                mStorageBuffers.push_back(vulkan_utils::createStorageBuffer(device.getDevice(),
                                                                            device.getMemoryProperties(),
                                                                            bufferContents.size()));
                mArgOrder.push_back(kind_storageBuffer);

                auto bufferMap = mStorageBuffers.back().map<void>();
                std::memcpy(bufferMap.get(), bufferContents.data(), bufferContents.size());
            }
            else if (*arg == "-ub") {
                // add a uniform buffer argument
                arg = std::next(arg);
                if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
                const auto bufferContents = hexToBytes(*arg);

                mUniformBuffers.push_back(vulkan_utils::createUniformBuffer(device.getDevice(),
                                                                            device.getMemoryProperties(),
                                                                            bufferContents.size()));
                mArgOrder.push_back(kind_uniformBuffer);

                auto bufferMap = mUniformBuffers.back().map<void>();
                std::memcpy(bufferMap.get(), bufferContents.data(), bufferContents.size());
            }
            else if (*arg == "-las") {
                // add a local array size argument
                arg = std::next(arg);
                if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");
                mLocalArraySizes.push_back(std::atoi(arg->c_str()));
                mArgOrder.push_back(kind_LocalArraySize);
            }
            else if (*arg == "-roi") {
                // TODO: support read only images
                clspv_utils::fail_runtime_error("generic does not yet support read-only images");
            }
            else if (*arg == "-woi") {
                // TODO: support write only images
                clspv_utils::fail_runtime_error("generic does not yet support write-only images");
            }
            else if (*arg == "-label") {
                // record a label for the test
                arg = std::next(arg);
                if (arg == args.end()) clspv_utils::fail_runtime_error("badly formed arguments to generic test");

                mParameterString = *arg;
            }
        }
    }

    void Test::prepare()
    {
    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        clspv_utils::invocation invocation(kernel.createInvocationReq());

        auto storageBufferArg = mStorageBuffers.begin();
        auto uniformBufferArg = mUniformBuffers.begin();
        auto localArraySizeArg = mLocalArraySizes.begin();

        for (auto arg : mArgOrder) {
            switch (arg)
            {
                case kind_storageBuffer:
                    if (storageBufferArg == mStorageBuffers.end()) clspv_utils::fail_runtime_error("not enough storage buffers");
                    invocation.addStorageBufferArgument(*storageBufferArg);
                    storageBufferArg = std::next(storageBufferArg);
                    break;

                case kind_uniformBuffer:
                    if (uniformBufferArg == mUniformBuffers.end()) clspv_utils::fail_runtime_error("not enough uniform buffers");
                    invocation.addUniformBufferArgument(*uniformBufferArg);
                    uniformBufferArg = std::next(uniformBufferArg);
                    break;

                case kind_LocalArraySize:
                    if (localArraySizeArg == mLocalArraySizes.end()) clspv_utils::fail_runtime_error("not enough local array sizes");
                    invocation.addLocalArraySizeArgument(*localArraySizeArg);
                    localArraySizeArg = std::next(localArraySizeArg);
                    break;

                default:
                    clspv_utils::fail_runtime_error("generic test encountered unrecognized arg type");
                    break;
            }
        }

        return invocation.run(mNumWorkgroups);
    }

    std::string Test::getParameterString() const
    {
        return mParameterString;
    }

    test_utils::Evaluation Test::evaluate(bool verbose)
    {
        test_utils::Evaluation result;
        result.mNumCorrect = 1;
        return result;
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        return test_utils::KernelTest::invocation_tests({ test_utils::make_invocation_test<Test>("") });
    }
}
