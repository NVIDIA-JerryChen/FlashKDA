#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("%s cc=%d.%d SMs=%d smem_optin=%zu clk=%d kHz\n", p.name, p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlockOptin, clk);
    return 0;
}
