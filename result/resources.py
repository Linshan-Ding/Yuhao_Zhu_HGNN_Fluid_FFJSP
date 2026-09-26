"""Local resource measurements, without optional process-monitor packages."""
import ctypes
import os


def memory_info():
    if os.name=='nt':
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(k,ctypes.c_ulonglong) for k in
                ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
        mem=Memory();mem.length=ctypes.sizeof(mem)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):raise OSError('Memory status unavailable')
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
                ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        psapi=ctypes.WinDLL('psapi');kernel=ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        count=Counters();count.cb=ctypes.sizeof(count)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(count),count.cb):raise OSError('Process memory unavailable')
        return dict(total_bytes=int(mem.total),available_bytes=int(mem.available),
                    peak_bytes=max(int(count.peak_working),int(count.peak_pagefile)))
    import resource
    from pathlib import Path
    fields={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return dict(total_bytes=fields['MemTotal'],available_bytes=fields['MemAvailable'],
                peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
