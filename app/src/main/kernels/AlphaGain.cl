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
    __write_only image2d_t  outDest,
    int                     inWidth,
    int                     inHeight,
    float                   inAlphaGainFactor)
{
    int x = KernelX();
    int y = KernelY();

    if (x < inWidth && y < inHeight)
    {
        float4 pixel = read_imagef(inImage, copyImageToBufferSampler, (float2)(x, y));
        pixel.a *= inAlphaGainFactor;
        write_imagef(outDest, (int2)(x, y), pixel);
    }
}

//__kernel void COMP_LUT3D(
//        __read_only image2d_t   inImage,
//        __read_only image3d_t   inLUT,
//        __write_only image2d_t  outDest,
//        int                     inWidth,
//        int                     inHeight)
//{
//    int x = KernelX();
//    int y = KernelY();
//
//    if (x < inWidth && y < inHeight)
//    {
//        float2 srcCoord = (float2)( ((float)x + 0.5f)/((float)inWidth),
//                                    ((float)y + 0.5f)/((float)inHeight) );
//        float4 pixel = read_imagef(inImage, linearSampler, srcCoord);
//        float3 lutCoord = (float2)( pixel.x, pixel.y, pixel.z);
//        float4 pixel = read_imagef(inLUT, linearSampler, lutCoord);
//
//        write_imagef(outDest, (int2)(x, y), pixel);
//    }
//}