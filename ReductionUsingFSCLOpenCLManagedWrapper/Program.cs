using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using OpenCL;

namespace ReductionUsingFSCLOpenCLManagedWrapper
{
    internal static class Program
    {
        private static void Main()
        {
            try
            {
                var platforms = OpenCLPlatform.Platforms.ToList();
                platforms.ForEach(platform =>
                {
                    Console.WriteLine($"{platform.Name}, {platform.Vendor}");
                    var devices = platform.Devices.ToList();
                    devices.ForEach(device =>
                    {
                        Console.WriteLine($"\t{device.Name}, {device.Vendor}");
                        RunKernel(platform, device);
                        Console.WriteLine();
                    });
                });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
            }
        }

        private static void RunKernel(OpenCLPlatform platform, OpenCLDevice device)
        {
            var context = new OpenCLContext(new List<OpenCLDevice> {device}, new OpenCLContextPropertyList(platform), null, IntPtr.Zero);
            var program = LoadProgram(context, device, "ReductionUsingFSCLOpenCLManagedWrapper.reduction.cl");
            var kernel1 = program.CreateKernel("reductionVector");
            var kernel2 = program.CreateKernel("reductionComplete");

            const int numValues = 1024 * 1024;
            const int numValuesPerWorkItem = 4;
            var globalWorkSize = numValues / numValuesPerWorkItem;
            const int localWorkSize = 32;
            var initialNumWorkGroups = globalWorkSize/localWorkSize;
            const int value = 42;
            var data = Enumerable.Repeat(value, numValues).Select(n => (float)n).ToArray();

            var commandQueue = new OpenCLCommandQueue(context, device, OpenCLCommandQueueProperties.None);

            var floatType = typeof (float);
            var floatSize = sizeof (float);

            var dataBuffer1 = new OpenCLBuffer(context, OpenCLMemoryFlags.ReadWrite | OpenCLMemoryFlags.AllocateHostPointer, floatType, new long[] {numValues});
            var dataBuffer2 = new OpenCLBuffer(context, OpenCLMemoryFlags.ReadWrite | OpenCLMemoryFlags.AllocateHostPointer, floatType, new long[] {initialNumWorkGroups*numValuesPerWorkItem});
            var sumBuffer = new OpenCLBuffer(context, OpenCLMemoryFlags.WriteOnly | OpenCLMemoryFlags.AllocateHostPointer, floatType, new long[] { 1 });
            var resultDataBuffer = dataBuffer2;

            using (var pinnedData = new PinnedObject(data))
            {
                commandQueue.WriteToBuffer(pinnedData, dataBuffer1, true, 0L, numValues);
            }

            foreach (var index in Enumerable.Range(0, int.MaxValue))
            {
                var dataBufferIn = index%2 == 0 ? dataBuffer1 : dataBuffer2;
                var dataBufferOut = index%2 == 0 ? dataBuffer2 : dataBuffer1;
                resultDataBuffer = dataBufferOut;

                kernel1.SetMemoryArgument(0, dataBufferIn);
                kernel1.SetMemoryArgument(1, dataBufferOut);
                kernel1.SetLocalArgument(2, localWorkSize*numValuesPerWorkItem*floatSize);

                Console.WriteLine($"Calling commandQueue.Execute(kernel1) with globalWorkSize: {globalWorkSize}; localWorkSize: {localWorkSize}; num work groups: {globalWorkSize / localWorkSize}");

                commandQueue.Execute(kernel1, null, new long[] {globalWorkSize}, new long[] {localWorkSize});

                globalWorkSize /= localWorkSize;
                if (globalWorkSize <= localWorkSize) break;
            }

            kernel2.SetMemoryArgument(0, resultDataBuffer);
            kernel2.SetLocalArgument(1, globalWorkSize*numValuesPerWorkItem*floatSize);
            kernel2.SetMemoryArgument(2, sumBuffer);

            Console.WriteLine($"Calling commandQueue.Execute(kernel2) with globalWorkSize: {globalWorkSize}; localWorkSize: {globalWorkSize}");

            commandQueue.Execute(kernel2, null, new long[] { globalWorkSize }, new long[] { globalWorkSize });

            commandQueue.Finish();

            var sum = new float[1];
            using (var pinnedSum = new PinnedObject(sum))
            {
                commandQueue.ReadFromBuffer(sumBuffer, pinnedSum, true, 0L, 1L);
            }

            const int correctAnswer = numValues * value;

            Console.WriteLine($"OpenCL final answer: {Math.Truncate(sum[0]):N0}; Correct answer: {correctAnswer:N0}");
        }

        private static OpenCLProgram LoadProgram(OpenCLContext context, OpenCLDevice device, string resourceName)
        {
            var source = GetProgramSourceFromResource(Assembly.GetExecutingAssembly(), resourceName);
            var program = new OpenCLProgram(context, source);

            try
            {
                program.Build(new List<OpenCLDevice> {device}, string.Empty, null, IntPtr.Zero);
            }
            catch (BuildProgramFailureOpenCLException)
            {
                var buildLog = program.GetBuildLog(device);
                throw new ApplicationException($"Error building program \"{resourceName}\":{Environment.NewLine}{buildLog}");
            }

            return program;
        }

        private static string GetProgramSourceFromResource(Assembly assembly, string resourceName)
        {
            using (var stream = assembly.GetManifestResourceStream(resourceName))
            {
                if (stream == null) throw new ApplicationException($"Failed to load resource {resourceName}");
                var streamReader = new StreamReader(stream);
                return streamReader.ReadToEnd();
            }
        }
    }
}
