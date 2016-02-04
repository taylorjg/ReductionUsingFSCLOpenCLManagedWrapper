using System;
using System.Runtime.InteropServices;

namespace ReductionUsingFSCLOpenCLManagedWrapper
{
    public class PinnedObject : IDisposable
    {
        private GCHandle _gcHandle;

        public PinnedObject(object value)
        {
            _gcHandle = GCHandle.Alloc(value, GCHandleType.Pinned);
        }

        public void Dispose()
        {
            _gcHandle.Free();
        }

        public static implicit operator IntPtr(PinnedObject pinnedObject)
        {
            return pinnedObject._gcHandle.AddrOfPinnedObject();
        }
    }
}
