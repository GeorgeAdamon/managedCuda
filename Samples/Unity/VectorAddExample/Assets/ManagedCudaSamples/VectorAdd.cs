using System;
using System.IO;
using  SD = System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = System.Random;
using ManagedCuda;
using ManagedCuda.BasicTypes;

using ManagedCuda.VectorTypes;
using Sirenix.OdinInspector;

/*
* This code is based on code from the NVIDIA CUDA SDK. (Ported from C++ to C# using managedCUDA)
* This software contains source code provided by NVIDIA Corporation.
*
*/

[ExecuteAlways]
public class VectorAdd : MonoBehaviour
{
    static CudaContext ctx;
    private static CudaKernel vectorAddKernel;
    static Random rand = new Random();

    public int Count;

    // C# Variables
    static float[] h_A;
    static float[] h_B;
    static float[] h_C;

    // CUDA Variables
    // Memory has to be copied here from host
    static CudaDeviceVariable<float> d_A;
    static CudaDeviceVariable<float> d_B;
    //static CudaDeviceVariable<float> d_C;
    
    // Shared Memory between host and GPU ( CUDA Unified Memory)
    //  static CudaManagedMemory_float A;
    //  static CudaManagedMemory_float B;
    static CudaManagedMemory_float C;

    public string resName =>Path.Combine(Application.dataPath, "ManagedCudaSamples", "vectorAdd_x64.ptx");


    private void OnEnable()
    { }

    [Button]
     public void GenerateRandomNumbers()
    {
        CleanupResources();
        
        //Init Cuda context
        ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());
        
        // Allocate input vectors h_A and h_B in host memory
        h_A = new float[Count];
        h_B = new float[Count];
        
        // Initialize input vectors
        RandomInit(h_A, Count);
        RandomInit(h_B, Count);
        
        // Allocate vectors in device memory and copy vectors from host memory to device memory 
        // Notice the new syntax with implicit conversion operators: Allocation of device memory and data copy is one operation.
        d_A = h_A;
        d_B = h_B;
        //d_C = new CudaDeviceVariable<float>(Count);
        
        // Allocate Shared Memory. The GPU will write here
        // A = new CudaManagedMemory_float(Count, CUmemAttach_flags.Global);
        // B = new CudaManagedMemory_float(Count, CUmemAttach_flags.Global);
       C = new CudaManagedMemory_float(Count, CUmemAttach_flags.Global);
        
    }
    
     
     
    [Button]
    public void CUDA_AddFloatArrays()
    {
        //Load Kernel image from resources
        Stream stream = new StreamReader(resName).BaseStream;
        if (stream == null) throw new ArgumentException("Kernel not found in resources.");

        vectorAddKernel = ctx.LoadKernelPTX(stream, "VecAdd");
        
        var threadsPerBlock = 1024;
        vectorAddKernel.BlockDimensions = threadsPerBlock;
        vectorAddKernel.GridDimensions = (Count + threadsPerBlock - 1) / threadsPerBlock;
        
        CudaStopWatch w = new CudaStopWatch();
        w.Start();
        vectorAddKernel.Run(d_A.DevicePointer,d_B.DevicePointer, C.DevicePointer, Count);
        w.Stop();
        
        Debug.Log(w.GetElapsedTime()/1000.0f);
        Debug.Log($"{h_A[0]} + {h_B[0]} = {C[0]}");
        Debug.Log($"{h_A[Count-1]} + {h_B[Count-1]} = {C[Count-1]}");
        
        // Copy result from device memory to host memory
        // h_C contains the result in host memory
        // h_C = d_C;
    }


    private void CleanupResources()
    {
        // Free device memory
        if (d_A != null) d_A?.Dispose();
        if (d_B != null) d_B?.Dispose();
        //   d_C?.Dispose();

        if (C != null) C?.Dispose();
        if (ctx != null) ctx?.Dispose();

        // Free host memory
        // We have a GC for that :-)
    }

    // Allocates an array with random float entries.
    private static void RandomInit(float[] data, int n)
    {
        for (int i = 0; i < n; ++i)
            
            data[i] = (float) rand.NextDouble();
    }
}