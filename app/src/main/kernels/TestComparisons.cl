__kernel void TestGreaterThanOrEqualTo(
    __global float* outDest,
    int             inWidth,
    int             inHeight)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int index = (y * inWidth) + x;

    if (x >= 0 && y >= 0 && x < inWidth && y < inWidth) {
        outDest[index] = 1.0f;
    }
    else {
        outDest[index] = 0.0f;
    }
}
