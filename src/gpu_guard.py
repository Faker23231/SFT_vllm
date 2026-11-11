# -*- coding: utf-8 -*-
"""训练/评测期间自动锁定 GPU 频率，规避 TDR 死机。

本机（RTX 5080 Laptop / 驱动 621.14）在出厂 boost 频率下不稳定：GPU 偶发挂起后
驱动等待超时触发 TDR 重置，重置失败即整机定格（事件日志 nvlddmkm 153 + Kernel-Power 41）。
实测把 SM 时钟锁到 300~1700 MHz 后温度、功耗下降，TDR 归零。

用法：
    from src.gpu_guard import GpuClockLock
    with GpuClockLock():      # 退出（含异常）自动恢复默认频率
        train()

    from src.gpu_guard import auto_lock
    auto_lock()               # 给没有 __main__ 保护的脚本用，注册 atexit 恢复

环境变量：
    SFT_GPU_CLOCK        锁频区间，形如 "300,1700"；设为 off 表示不锁（默认 300,2000）
    SFT_GPU_CLOCK_KEEP   设为 1 时结束后不恢复默认频率（默认恢复）

设置时钟需要管理员权限（Windows）/ root（Linux）；非管理员只打印提示，不中断训练。
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess

DEFAULT_CLOCK = "300,2000"  # 下限 300 MHz 允许空闲降频，上限留出稳定余量
_OFF_VALUES = {"", "off", "none", "false", "0"}


def _is_admin() -> bool:
    if os.name != "nt":
        try:
            return os.geteuid() == 0
        except AttributeError:
            return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if os.name == "nt":
        for path in (
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ):
            if os.path.exists(path):
                return path
    return None


def _run(args: list[str]) -> subprocess.CompletedProcess:
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW，避免弹控制台窗口
    return subprocess.run(args, capture_output=True, text=True, timeout=60, creationflags=flags)


def _current_sm_clock(exe: str) -> str:
    try:
        result = _run([exe, "-i", "0", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
        return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        return "?"


def lock_gpu_clocks(clock_range: str | None = None, verbose: bool = True) -> bool:
    """锁定 SM 时钟。clock_range 优先于 SFT_GPU_CLOCK；返回是否真的锁上。"""
    spec = (clock_range if clock_range is not None else os.environ.get("SFT_GPU_CLOCK", DEFAULT_CLOCK)).strip()
    if spec.lower() in _OFF_VALUES:
        if verbose:
            print("[gpu-guard] SFT_GPU_CLOCK=off，跳过锁频")
        return False

    exe = _nvidia_smi()
    if exe is None:
        if verbose:
            print("[gpu-guard] 未找到 nvidia-smi，跳过锁频")
        return False

    low, _, high = spec.partition(",")  # 支持 "1700"（只给上限）和 "300,1700" 两种写法
    low = low.strip() or "0"
    high = high.strip() or low

    try:
        result = _run([exe, "-i", "0", "-lgc", f"{low},{high}"])  # -i 0 指定第一块卡
    except Exception as exc:
        if verbose:
            print(f"[gpu-guard] 调用 nvidia-smi 失败：{exc}")
        return False

    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        text = output.lower()
        if any(key in text for key in ("permission", "admin", "insufficient", "拒绝", "权限")):
            print("[gpu-guard] 锁频需要管理员权限：请用『以管理员身份运行』的终端/IDE 启动训练，"
                  "或改用 nvidia-smi -lgc 的开机计划任务")
        elif "not supported" in text or "不支持" in output:
            print("[gpu-guard] 当前 vBIOS/驱动不允许锁频，请在 Armoury Crate 里降低 GPU 频率档位")
        else:
            print(f"[gpu-guard] 锁频失败：{output}")
        return False

    if verbose:
        print(f"[gpu-guard] 已锁定 SM 时钟 {low}~{high} MHz（当前 {_current_sm_clock(exe)} MHz）")
    return True


def unlock_gpu_clocks(verbose: bool = True) -> bool:
    exe = _nvidia_smi()
    if exe is None:
        return False
    try:
        result = _run([exe, "-i", "0", "-rgc"])  # -rgc = reset gpu clocks
    except Exception:
        return False
    ok = result.returncode == 0
    if verbose:
        print("[gpu-guard] 已恢复默认频率（-rgc）" if ok else "[gpu-guard] 恢复默认频率失败")
    return ok


def auto_lock(clock_range: str | None = None, verbose: bool = True) -> bool:
    """锁频并注册 atexit 恢复；返回值只表示本次是否锁上，失败不影响后续代码。"""
    locked = lock_gpu_clocks(clock_range, verbose)
    if locked:
        atexit.register(unlock_gpu_clocks, verbose)
    return locked


class GpuClockLock:
    """上下文管理器：进入时锁频，退出（含异常）时恢复。"""

    def __init__(
        self,
        clock_range: str | None = None,
        keep_after: bool | None = None,
        verbose: bool = True,
    ) -> None:
        self.clock_range = clock_range
        self.verbose = verbose
        self.keep_after = (
            keep_after if keep_after is not None else os.environ.get("SFT_GPU_CLOCK_KEEP", "0") == "1"
        )
        self.locked = False  # 没锁成功就不去调 -rgc，避免打乱用户自己设的计划任务

    def __enter__(self) -> bool:
        self.locked = lock_gpu_clocks(self.clock_range, self.verbose)
        return self.locked

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.locked and not self.keep_after:
            unlock_gpu_clocks(self.verbose)
        return False  # 不吞异常
