To build a program in any of 01_CUDA/* :

Pre-requisites:
a. You must have an NVIDIA GPU
b. It must be CUDA-Enabled (refer to this link for more info a CUDA-Enabled GPU: https://developer.nvidia.com/cuda-gpus)
c. You must have the Nvidia GPU Computing Toolkit installed

1. Open a command prompt instance
2. Navigate to the program you want to build in the prompt
3. Issue the following command (replace source_file.cu with the name of the program in the prompt's current directory and exec_file.exe by the name you want the executable to have):

   > nvcc.exe -o exec_file.exe source_file.cu

4. Now run exec_file.exe by issuing the following command:

   > exec_file.exe

##########################################################

To build a program in any of 02_OpenCL/* :

Pre-requisites:
a. You must have OpenCL-supported GPU Platform with the OpenCL SDK supplied by GPU Vendor installed.
   To see if this satisfies, try running the DepProp.exe program (located at 02_OpenCL/04_DeviceProperties). If you see information about a supporting GPU Device, then you are good to go!

1. Open a command prompt instance
2. Navigate to the program you want to build in the prompt
3. Issue the following commands (replace source_file.c/cpp with the name of the program in the prompt's current directory, ... is where your NVIDIA GPU Computing Toolkit directory resides):

   > cl.exe /c /EHsc /I "...\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\include" source_file.c
   > link.exe source_file.obj "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64\OpenCL.lib" /MACHINE:x64 /SUBSYSTEM:CONSOLE

4. Now run exec_file.exe by issuing the following command:

   > source_file.exe

===========================================================
- Kaivalya Deshpande
===========================================================
