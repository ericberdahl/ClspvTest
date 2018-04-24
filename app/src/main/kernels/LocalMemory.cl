/*
 * This is an odd algorithm that fundamentally is equivalent to:
 *    outDest[i] = inSource[ inIndices[i] ];
 *
 * The obfuscation is because we want to test dynamic local store in clspv (the __local argument to
 * the kernel. Thus, the kernel actually stages indices from inIndices into sharedIndices, and does
 * so in a way that should be relatively difficult for a compiler to optimize through.
 */
__kernel void StrangeShuffle(
    __global int*       inIndices,
    __global float4*    inSource,
    __global float4*    outDest,
    __local int*        sharedIndices)
{
    // collect information about the current work item's location in the data space
    uint group_id   = get_group_id(0);
    uint local_size = get_local_size(0);
    uint local_id   = get_local_id(0);

    // Step 1. Copy indices from inIndices into sharedIndices.

    uint localIndexWriteBase = local_size - local_id - 1;
    uint globalReadBase = 2*local_size*group_id + localIndexWriteBase;

    sharedIndices[localIndexWriteBase             ] = inIndices[globalReadBase             ];
    sharedIndices[localIndexWriteBase + local_size] = inIndices[globalReadBase + local_size];

    // Step 2. Wait for all items in the work group to finish copying indices

    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3. Use the values in sharedIndices to copy from inSource to outDest

    uint localIndexReadBase = 2*local_id;
    uint globalWriteBase = 2*local_size*group_id + localIndexReadBase;

    outDest[globalWriteBase    ] = inSource[ sharedIndices[localIndexReadBase    ] ];
    outDest[globalWriteBase + 1] = inSource[ sharedIndices[localIndexReadBase + 1] ];
}

