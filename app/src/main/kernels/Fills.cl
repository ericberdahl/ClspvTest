int KernelX()
{
    return get_global_id(0);
}

int KernelY()
{
    return get_global_id(1);
}

void WriteFloat4(
    float4              inPixel,
    __global float4*    outImage,
    int                 inIndex,
    bool                is16Bit)
{
    if (is16Bit)
    {
        vstore_half4_rtz(inPixel, inIndex, (__global half*)outImage);
    }
    else
    {
        outImage[inIndex] = inPixel;
    }
}

void WritePixelIndex(
    float4              inPixel,
    __global float4*    outImage,
    bool                is16Bit,
    int                 inIndex)
{
    WriteFloat4(inPixel, outImage, inIndex, is16Bit);
}

void WritePixel(
    float4              inPixel,
    __global float4*    outImage,
    int                 inPitch,
    bool                is16Bit,
    int                 inX,
    int                 inY)
{
    WritePixelIndex(inPixel, outImage, is16Bit, mul24(inY, inPitch) + inX);
}

__kernel void FillWithColorKernel(
    __global float4*    outImage,
    int                 inPitch,
    int                 inDeviceFormat,
    int                 inOffsetX,
    int                 inOffsetY,
    int                 inWidth,
    int                 inHeight,
    float4              inColor)
{
    int x = KernelX();
    int y = KernelY();

    if (x < inWidth && y < inHeight)
    {
        WritePixel(inColor, outImage, inPitch, inDeviceFormat == 0, x + inOffsetX, y + inOffsetY);
    }
}

