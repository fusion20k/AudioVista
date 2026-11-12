"""Windows taskbar integration (no third-party packages, just ctypes).

Gives the AudioVista window its own taskbar identity so Windows shows the
AudioVista logo instead of the Python icon. Setting the ID on the window's
property store (rather than only per process) is needed because the
Microsoft Store build of Python is a packaged app, and Windows otherwise
groups its windows under Python's own icon.
"""

from __future__ import annotations

import ctypes
import io
import struct
import sys
from ctypes import wintypes
from pathlib import Path

import pygame

VT_LPWSTR = 31


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> "_GUID":
        h = text.strip("{}").replace("-", "")
        return cls(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16),
                   (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:32])))


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("pwszVal", ctypes.c_wchar_p), ("pad", ctypes.c_void_p)]


_IID_IPROPERTYSTORE = "{886d8eeb-8cf2-4446-8d02-cdba1dbdcf99}"
_APPUSERMODEL = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
PKEY_RELAUNCH_COMMAND = 2
PKEY_RELAUNCH_ICON = 3
PKEY_RELAUNCH_NAME = 4
PKEY_ID = 5


def set_window_identity(hwnd: int, app_id: str, display_name: str = "",
                        icon_path: str | None = None, relaunch_command: str | None = None) -> bool:
    """Assign a taskbar AppUserModelID (plus relaunch info for pinning) to *hwnd*."""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        shell32 = ctypes.windll.shell32
        store = ctypes.c_void_p()
        iid = _GUID.parse(_IID_IPROPERTYSTORE)
        if shell32.SHGetPropertyStoreForWindow(wintypes.HWND(hwnd), ctypes.byref(iid),
                                               ctypes.byref(store)) != 0 or not store:
            return False
        vtbl = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        proto = ctypes.WINFUNCTYPE
        set_value = proto(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_PROPERTYKEY),
                          ctypes.POINTER(_PROPVARIANT))(vtbl[6])
        commit = proto(ctypes.c_long, ctypes.c_void_p)(vtbl[7])
        release = proto(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])

        def put(pid: int, value: str):
            key = _PROPERTYKEY(_GUID.parse(_APPUSERMODEL), pid)
            var = _PROPVARIANT(VT_LPWSTR, 0, 0, 0, value, None)
            set_value(store, ctypes.byref(key), ctypes.byref(var))

        # Relaunch properties must be set before the ID.
        if relaunch_command:
            put(PKEY_RELAUNCH_COMMAND, relaunch_command)
        if icon_path:
            put(PKEY_RELAUNCH_ICON, icon_path)
        if display_name:
            put(PKEY_RELAUNCH_NAME, display_name)
        put(PKEY_ID, app_id)
        commit(store)
        release(store)
        return True
    except (AttributeError, OSError, ValueError):
        return False


def write_ico(path: str | Path, images: list[pygame.Surface]) -> Path:
    """Write a multi-size .ico file (PNG-compressed entries, Windows Vista+)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs = []
    for img in images:
        buf = io.BytesIO()
        pygame.image.save(img, buf, "icon.png")
        blobs.append((img.get_width(), img.get_height(), buf.getvalue()))
    header = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries, data = b"", b""
    for w, h, png in blobs:
        entries += struct.pack("<BBBBHHII", w % 256, h % 256, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    path.write_bytes(header + entries + data)
    return path


def set_window_icon(hwnd: int, ico_path: str | Path) -> bool:
    """Load a multi-size .ico and give the window both its small and large icons,
    so Windows can pick a sharp size for the title bar, taskbar and Alt+Tab."""
    if sys.platform != "win32" or not hwnd or not Path(ico_path).is_file():
        return False
    try:
        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                      ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON = 1, 0x10, 0x0080
        ok = False
        for which, metric_w, metric_h in ((0, 49, 50), (1, 11, 12)):  # ICON_SMALL / ICON_BIG
            w, h = user32.GetSystemMetrics(metric_w), user32.GetSystemMetrics(metric_h)
            icon = user32.LoadImageW(None, str(ico_path), IMAGE_ICON, w, h, LR_LOADFROMFILE)
            if icon:
                user32.SendMessageW(hwnd, WM_SETICON, which, icon)
                ok = True
        return ok
    except (AttributeError, OSError):
        return False
