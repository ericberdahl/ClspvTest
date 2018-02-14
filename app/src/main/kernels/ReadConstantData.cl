int KernelX()
{
    return get_global_id(0);
}

// Lookup table for the function 2^x, defined on the domain [0,9]
__constant float kTwoPowX[10] =
{
    1.0f,
    2.0f,
    4.0f,
    8.0f,
    16.0f,
    32.0f,
    64.0f,
    128.0f,
    256.0f,
    512.0f
};

__kernel void ReadConstantData(__global float* outDest, int inWidth)
{
    int index = KernelX();

    if (index < min((size_t)inWidth, (sizeof(kTwoPowX) / sizeof(kTwoPowX[0])))) {
        outDest[index] = kTwoPowX[index];
    }
    else {
        outDest[index] = -1.0f;
    }
}
