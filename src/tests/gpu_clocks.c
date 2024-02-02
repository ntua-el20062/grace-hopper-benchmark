#include <stdio.h>
#include <nvml.h>

int main()
{
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    unsigned int graphicsClock, memoryClock, smClock, videoClock;
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graphicsClock);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &memoryClock);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &smClock);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_VIDEO, &videoClock);

    printf("GPU 0 - Graphics Clock: %u MHz, Memory Clock: %u MHz\n", graphicsClock, memoryClock);
    printf("GPU 0 - SM Clock: %u MHz, Video Clock: %u MHz\n", smClock, videoClock);

    nvmlShutdown();
    return 0;
}
