#include <cecl.h>
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// OpenGL Graphics Includes
#include <GL/glew.h>
#ifdef UNIX
    #include <GL/glxew.h>
#endif
#if defined (_WIN32)
    #include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif

//Standard utilities and systems includes
#include <oclUtils.h>

#include "particleSystem_common.h"
#include "particleSystem_engine.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for particles kernels
////////////////////////////////////////////////////////////////////////////////
//OpenCL particles program
static cl_program cpParticles;

//OpenCL particles kernels
static cl_kernel
    ckIntegrate,
    ckCalcHash,
    ckMemset,
    ckFindCellBoundsAndReorder,
    ckCollide;

//Default command queue for particles kernels
static cl_command_queue cqDefaultCommandQue;

//Simulation parameters
static cl_mem params;

static size_t wgSize = 64;

extern "C" void initParticles(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog("Loading Particles.cl...\n");
        char *cParticles = oclLoadProgSource(shrFindFilePath("Particles.cl", argv[0]), "// My comment\n", &kernelLength);
        oclCheckError(cParticles != NULL, shrTRUE);

    shrLog("Creating particles program...\n");
        cpParticles = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cParticles, &kernelLength, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Building particles program...\n");
        ciErrNum = CECL_PROGRAM(cpParticles, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			// write out standard error, Build Log and PTX, then cleanup and exit
			shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
			oclLogBuildInfo(cpParticles, oclGetFirstDev(cxGPUContext));
			oclLogPtx(cpParticles, oclGetFirstDev(cxGPUContext), "oclParticles.ptx");
			oclCheckError(ciErrNum, CL_SUCCESS); 
		}

    shrLog("Creating particles kernels...\n\n");
        ckIntegrate = CECL_KERNEL(cpParticles, "integrate", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckCalcHash = CECL_KERNEL(cpParticles, "calcHash", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckMemset = CECL_KERNEL(cpParticles, "Memset", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckFindCellBoundsAndReorder = CECL_KERNEL(cpParticles, "findCellBoundsAndReorder", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        ckCollide = CECL_KERNEL(cpParticles, "collide", &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Creating parameter GPU buffer...\n\n");
        allocateArray(&params, sizeof(simParams_t));

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cParticles);

    //Save ptx code to separate file
    oclLogPtx(cpParticles, oclGetFirstDev(cxGPUContext), "Particles.ptx");
}

extern "C" void closeParticles(void){
    cl_int ciErrNum;
    ciErrNum  = clReleaseMemObject(params);
    ciErrNum |= clReleaseKernel(ckCollide);
    ciErrNum |= clReleaseKernel(ckFindCellBoundsAndReorder);
    ciErrNum |= clReleaseKernel(ckMemset);
    ciErrNum |= clReleaseKernel(ckCalcHash);
    ciErrNum |= clReleaseKernel(ckIntegrate);
    ciErrNum |= clReleaseProgram(cpParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void setParameters(simParams_t *m_params){
    copyArrayToDevice(params, m_params, 0, sizeof(simParams_t));
}

static size_t uSnap(size_t a, size_t b){
    return ((a % b) == 0) ? a : (a - (a % b) + b);
}

extern "C" void integrateSystem(
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    float deltaTime,
    uint numParticles
){
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum  = CECL_SET_KERNEL_ARG(ckIntegrate, 0, sizeof(cl_mem), (void *)&d_Pos);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckIntegrate, 1, sizeof(cl_mem), (void *)&d_Vel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckIntegrate, 2, sizeof(cl_mem), (void *)&params);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckIntegrate, 3, sizeof(float), (void *)&deltaTime);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckIntegrate, 4, sizeof(uint), (void *)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqDefaultCommandQue, ckIntegrate, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void calcHash(
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    int numParticles
){
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum  = CECL_SET_KERNEL_ARG(ckCalcHash, 0, sizeof(cl_mem), (void *)&d_Hash);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCalcHash, 1, sizeof(cl_mem), (void *)&d_Index);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCalcHash, 2, sizeof(cl_mem), (void *)&d_Pos);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCalcHash, 3, sizeof(cl_mem), (void *)&params);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCalcHash, 4,  sizeof(uint), (void *)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqDefaultCommandQue, ckCalcHash, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

static void memsetOCL(
    memHandle_t d_Data,
    uint val,
    uint N
){
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(N, wgSize);

    ciErrNum  = CECL_SET_KERNEL_ARG(ckMemset, 0, sizeof(cl_mem), (void *)&d_Data);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckMemset, 1, sizeof(cl_uint), (void *)&val);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckMemset, 2, sizeof(cl_uint), (void *)&N);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqDefaultCommandQue, ckMemset, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void findCellBoundsAndReorder(
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    uint numParticles,
    uint numCells
){
    cl_int ciErrNum;
    memsetOCL(d_CellStart, 0xFFFFFFFFU, numCells);
    //memsetOCL(d_CellEnd, 0xFFFFFFFFU, numCells);
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum  = CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 0, sizeof(cl_mem), (void *)&d_CellStart);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 1, sizeof(cl_mem), (void *)&d_CellEnd);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 2, sizeof(cl_mem), (void *)&d_ReorderedPos);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 3, sizeof(cl_mem), (void *)&d_ReorderedVel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 4, sizeof(cl_mem), (void *)&d_Hash);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 5, sizeof(cl_mem), (void *)&d_Index);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 6, sizeof(cl_mem), (void *)&d_Pos);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 7, sizeof(cl_mem), (void *)&d_Vel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 8, (wgSize + 1) * sizeof(cl_uint), NULL);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckFindCellBoundsAndReorder, 9, sizeof(cl_uint), (void *)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqDefaultCommandQue, ckFindCellBoundsAndReorder, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void collide(
    memHandle_t d_Vel,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Index,
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    uint   numParticles,
    uint   numCells
){
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum  = CECL_SET_KERNEL_ARG(ckCollide, 0, sizeof(cl_mem), (void *)&d_Vel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 1, sizeof(cl_mem), (void *)&d_ReorderedPos);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 2, sizeof(cl_mem), (void *)&d_ReorderedVel);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 3, sizeof(cl_mem), (void *)&d_Index);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 4, sizeof(cl_mem), (void *)&d_CellStart);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 5, sizeof(cl_mem), (void *)&d_CellEnd);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 6, sizeof(cl_mem), (void *)&params);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckCollide, 7, sizeof(uint),   (void *)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = CECL_ND_RANGE_KERNEL(cqDefaultCommandQue, ckCollide, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}
