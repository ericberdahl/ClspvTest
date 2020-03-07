int KernelX()
{
    return get_global_id(0);
}

int KernelY()
{
    return get_global_id(1);
}

const sampler_t linearSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t copyImageToBufferSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void AlphaGainKernel(
    __read_only image2d_t   inImage,
    __global float4*        outBuffer,
    int                     inPitch,
    int                     inDeviceFormat,
    int                     inWidth,
    int                     inHeight,
    float                   inAlphaGainFactor)
{
    int x = KernelX();
    int y = KernelY();

    if (x < inWidth && y < inHeight)
    {
        float4 pixel = read_imagef(inImage, copyImageToBufferSampler, (float2)(x, y));
        pixel.w *= inAlphaGainFactor;
        int dstIndex = mul24(y, inPitch) + x;
        outBuffer[dstIndex] = pixel;
    }
}