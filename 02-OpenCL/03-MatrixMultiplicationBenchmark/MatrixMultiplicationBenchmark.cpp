/*
 * Authored by: Kaivalya Deshpande
 */

/* header files */
// standard headers
#include <stdlib.h>
#include <stdio.h>

// OpenCL headers
#include <CL/opencl.h>

// misc. headers
#include "include/helper_timer.h"	// for time

/* macros */
#define BLOCK_SIZE 1024

/* global variables */
cl_platform_id oclPlatformId;
cl_device_id oclDeviceId;

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

float *host_A = NULL;
float *host_B = NULL;
float *host_C = NULL;
float *gold = NULL;

cl_mem device_A = NULL;
cl_mem device_B = NULL;
cl_mem device_C = NULL;

float timeOnGPU = 0.0f;
float timeOnCPU = 0.0f;

const char *oclSourceCode =
"__kernel void multMat_GPU(__global float *A, __global float *B, __global float *C, int num_rows_A, int num_cols_A, int num_cols_B, int num_cols_C)" \
"{"                                                                                            \
	"int row = get_global_id(0);"                                                              \
	"int col = get_global_id(1);"                                                              \
	"float sum = 0;"                                                                           \
                                                                                               \
	"if (row < num_rows_A && col < num_cols_B)"                                                \
	"{"                                                                                        \
		"for(int k = 0; k < num_cols_A; k++)"                                                  \
		"{"                                                                                    \
			"sum = sum + ((*(A + (row * num_cols_A) + k)) * (*(B + (k * num_cols_B) + col)));" \
		"}"                                                                                    \
                                                                                               \
		"*(C + (row * num_cols_C) + col) = sum;"                                               \
	"}"                                                                                        \
"}";

/* entry-point function */
int main(void)
{
	// function prototypes
	void cleanup(void);
	void fillMatrix(float *const, size_t, size_t);
	void multMat_CPU(const float *const, const float *const, float *const, size_t, size_t, size_t, size_t);

	// variable declarations
	const size_t num_rows_A = BLOCK_SIZE;
	const size_t num_cols_A = BLOCK_SIZE;

	const size_t num_rows_B = BLOCK_SIZE;
	const size_t num_cols_B = BLOCK_SIZE;
	
	const size_t num_rows_C = num_rows_A;
	const size_t num_cols_C = num_cols_B;

	const size_t num_rows_gold = num_rows_C;
	const size_t num_cols_gold = num_cols_C;
	
	const size_t size_A = num_rows_A * num_cols_A * sizeof(float);
	const size_t size_B = num_rows_B * num_cols_B * sizeof(float);
	const size_t size_C = num_rows_C * num_cols_C * sizeof(float);
	const size_t size_gold = num_rows_gold * num_cols_gold * sizeof(float);

	cl_int result;

	// code
	// allocate host memory
	host_A = (float *)malloc(size_A);
	if (!host_A)
	{
		printf("malloc: failed to allocate memory for host_A\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	host_B = (float *)malloc(size_B);
	if (!host_B)
	{
		printf("malloc: failed to allocate memory for host_B\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	host_C = (float *)malloc(size_C);
	if (!host_C)
	{
		printf("malloc: failed to allocate memory for host_C\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (float *)malloc(size_gold);
	if (!gold)
	{
		printf("malloc: failed to allocate memory for gold\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// display matrix dimensions
	printf("\n");
	printf("Dimensions of Matrix A: %zd x %zd\n", num_rows_A, num_cols_A);
	printf("Dimensions of Matrix B: %zd x %zd\n", num_rows_B, num_cols_B);
	printf("Dimensions of Matrix C: %zd x %zd\n", num_rows_C, num_cols_C);
	printf("Dimensions of gold    : %zd x %zd\n\n", num_rows_gold, num_cols_gold);

	// display matrix sizes in bytes
	printf("Size of Matrix A: %zd bytes\n", size_A);
	printf("Size of Matrix B: %zd bytes\n", size_B);
	printf("Size of Matrix C: %zd bytes\n", size_C);
	printf("Size of gold    : %zd bytes\n\n", size_gold);

	// populate host input matrices
	fillMatrix(host_A, num_rows_A, num_cols_A);
	fillMatrix(host_B, num_rows_B, num_cols_B);

	// obtain the 1st OpenCL supporting platform's ID
	result = clGetPlatformIDs(1, &oclPlatformId, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clGetPlatformIDs: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// obtain the 1st OpenCL supporting device's ID
	result = clGetDeviceIDs(oclPlatformId, CL_DEVICE_TYPE_GPU, 1, &oclDeviceId, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clGetDeviceIDs: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// create an OpenCL compute context
	oclContext = clCreateContext(NULL, 1, &oclDeviceId, NULL, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateContext: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// create command queue
	oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceId, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateCommandQueue: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	// create program from source
	oclProgram = clCreateProgramWithSource(oclContext, 1, &oclSourceCode, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateProgramWithSource: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// build
	result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		size_t len;
		char build_log[2048];

		result = clGetProgramBuildInfo(oclProgram, oclDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &len);
		if (result != CL_SUCCESS)
		{
			printf("clGetProgramBuildInfo: failed with code %d\n", result);
			cleanup();
			exit(EXIT_FAILURE);
		}

		printf("build error:\n");
		printf("build log for 'oclSourceCode':\n");
		printf("%s\n", build_log);

		printf("clBuildProgram: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// create OpenCL kernel
	oclKernel = clCreateKernel(oclProgram, "multMat_GPU", &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateKernel: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// allocate device memory
	device_A = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size_A, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer: failed to allocate device memory for device_A with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	device_B = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size_B, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer: failed to allocate device memory for device_B with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	device_C = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size_C, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer: failed to allocate device memory for device_C with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// set the arguments to pass
	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&device_A);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 0 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&device_B);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 1 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&device_C);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 2 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&num_rows_A);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 3 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void *)&num_cols_A);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 4 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void *)&num_cols_B);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 5 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void *)&num_cols_C);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArgs: failed for argument 6 with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// populate device input buffers
	result = clEnqueueWriteBuffer(oclCommandQueue, device_A, CL_FALSE, 0, size_A, host_A, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer: failed for device_A with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clEnqueueWriteBuffer(oclCommandQueue, device_B, CL_FALSE, 0, size_B, host_B, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer: failed for device_B with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// setting up the timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);

	// start timer
	sdkStartTimer(&timer);
	{
		// configure the kernel
		const size_t global_work_size[2] = { BLOCK_SIZE, BLOCK_SIZE };

		result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
		if (result != CL_SUCCESS)
		{
			printf("clEnqueueNDRangeKernel: failed with code %d\n", result);

			sdkStopTimer(&timer);
			sdkDeleteTimer(&timer);
			timer = NULL;

			cleanup();
			exit(EXIT_FAILURE);
		}

		// execute
		clFinish(oclCommandQueue);
	}
	// stop timer
	sdkStopTimer(&timer);

	timeOnGPU = sdkGetTimerValue(&timer);

	sdkDeleteTimer(&timer);
	timer = NULL;

	// read results back into host buffer
	result = clEnqueueReadBuffer(oclCommandQueue, device_C, CL_TRUE, 0, size_C, host_C, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueReadBuffer: failed with code %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// reset timer
	sdkCreateTimer(&timer);
	
	// start timer
	sdkStartTimer(&timer);
	{
		multMat_CPU(host_A, host_B, gold, num_rows_A, num_cols_A, num_cols_B, num_cols_C);
	}
	sdkStopTimer(&timer);

	timeOnCPU = sdkGetTimerValue(&timer);

	sdkDeleteTimer(&timer);
	timer = NULL;

	// comparison for accuracy
	const float epsilon = 0.000001f;
	int firstInaccurateIndex[2] = {-1, -1};
	bool isAccurate = true;

	for (int row = 0; row < num_rows_gold; row++)
	{
		for (int col = 0; col < num_cols_gold; col++)
		{
			if (fabs(gold[(row * num_rows_gold) + col] - host_C[(row * num_rows_C) + col]) > epsilon)
			{
				isAccurate = false;
				firstInaccurateIndex[0] = row;
				firstInaccurateIndex[1] = col;
				break;
			}
		}
	}
	
	char statementOfAccuracy[128];
	if (!isAccurate)
	{
		sprintf(statementOfAccuracy, "GPU produced atleast 1 result not within %.6f of that produced by the CPU at index (%d, %d)", epsilon, firstInaccurateIndex[0], firstInaccurateIndex[1]);
	}
	else
	{
		sprintf(statementOfAccuracy, "All GPU results are within %.6f of CPU results", epsilon);
	}

	// comparison for performance
	float timeRatio = timeOnCPU / timeOnGPU;

	char statementOfPerformance[64];
	if (timeRatio < 1.0f)
	{
		sprintf(statementOfPerformance, "CPU performed %.6f times faster than the GPU", 1.0f / timeRatio);
	}
	else
	{
		sprintf(statementOfPerformance, "GPU performed %.6f times faster than the CPU", timeRatio);
	}

	// display results
	printf("\tTime spent on the CPU: %.6f ms\n\n", timeOnCPU);

	printf("\tTime spent on the GPU: %.6f ms\n\n", timeOnGPU);

	printf("CPU vs GPU with respect to Accuracy:\n");
	printf("\t%s\n", statementOfAccuracy);

	printf("CPU vs GPU with respect to performance:\n");
	printf("\t%s\n", statementOfPerformance);

	// cleanup
	cleanup();

	return 0;
}

void fillMatrix(float *const mat, size_t m, size_t n)
{
	// variable declarations
	const float scalar = 1.0f / (float)RAND_MAX;

	// code
	srand(time(NULL));
	for (int i = 0; i < (m * n); i++)
		*(mat + i) = scalar * rand();
}

void multMat_CPU(const float *const A, const float *const B, float *const C, size_t num_rows_A, size_t num_cols_A, size_t num_cols_B, size_t num_cols_C)
{
	// variable declarations
	int row, col, sum = 0;

	// code
	for (row = 0; row < num_rows_A; row++)
	{
		for (col = 0; col < num_cols_B; col++)
		{
			for (int k = 0; k < num_cols_A; k++)
			{
				sum = sum + ((*(A + (row * num_cols_A) + k)) * (*(B + (k * num_cols_B) + col)));
			}
			*(C + (row * num_cols_C) + col) = sum;
		}
	}
}

void cleanup(void)
{
	// code
	if (device_C)
	{
		clReleaseMemObject(device_C);
		device_C = NULL;
	}
	if (device_B)
	{
		clReleaseMemObject(device_B);
		device_B = NULL;
	}
	if (device_A)
	{
		clReleaseMemObject(device_A);
		device_A = NULL;
	}
	if (oclKernel)
	{
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}
	if (oclProgram)
	{
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}
	if (oclCommandQueue)
	{
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}
	if (oclContext)
	{
		clReleaseContext(oclContext);
		oclContext = NULL;
	}
	if (gold)
	{
		free(gold);
		gold = NULL;
	}
	if (host_C)
	{
		free(host_C);
		host_C = NULL;
	}
	if (host_B)
	{
		free(host_B);
		host_B = NULL;
	}
	if (host_A)
	{
		free(host_A);
		host_A = NULL;
	}
}
