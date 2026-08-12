import yaml
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load config.yaml: {e}")
    else:
        print(f"[WARN] config.yaml not found at {CONFIG_PATH}")
    return {}

APP_CONFIG = load_config()

def get_path(key: str, default_relative: str = "") -> Path:
    """
    Get a resolved absolute Path object for a given config key.
    If the path specified in config (or default) is relative, it will be resolved relative to PROJECT_ROOT.
    """
    raw_path = APP_CONFIG.get("paths", {}).get(key)
    if not raw_path:
        raw_path = default_relative
    p = Path(raw_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p

_CPU_NAME_CACHE = None
_GPU_NAME_CACHE = None

def get_cpu_name() -> str:
    """Detect CPU model name across Windows, Linux, and macOS."""
    global _CPU_NAME_CACHE
    if _CPU_NAME_CACHE is not None:
        return _CPU_NAME_CACHE
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if name:
                _CPU_NAME_CACHE = name.strip()
                return _CPU_NAME_CACHE
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        _CPU_NAME_CACHE = line.split(":")[1].strip()
                        return _CPU_NAME_CACHE
        elif platform.system() == "Darwin":
            import subprocess
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                _CPU_NAME_CACHE = res.stdout.strip()
                return _CPU_NAME_CACHE
    except Exception:
        pass
    proc = platform.processor()
    _CPU_NAME_CACHE = proc if proc else "CPU"
    return _CPU_NAME_CACHE

def get_jetson_model() -> str:
    """Detect NVIDIA Jetson board model from device tree if available on Linux."""
    try:
        dt_model_path = Path("/proc/device-tree/model")
        if dt_model_path.exists():
            with open(dt_model_path, "r", encoding="utf-8", errors="ignore") as f:
                model_str = f.read().strip().rstrip('\x00')
                if model_str:
                    return model_str
    except Exception:
        pass
    return ""

def get_gpu_name() -> str:
    """Detect GPU device name dynamically using PyTorch & Jetson hardware info."""
    global _GPU_NAME_CACHE
    if _GPU_NAME_CACHE is not None:
        return _GPU_NAME_CACHE
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).strip()
            
            # Check Jetson device tree model first
            jetson_model = get_jetson_model()
            if jetson_model:
                clean_jetson = jetson_model.replace("Developer Kit", "").replace("NVIDIA ", "").strip()
                _GPU_NAME_CACHE = clean_jetson
                return _GPU_NAME_CACHE
            
            # If PyTorch returns "Orin" or "Orin (nvgpu)", expand to "Jetson AGX Orin"
            if name.lower() in ["orin", "orin (nvgpu)", "nvgpu"]:
                name = "Jetson AGX Orin"
            elif "orin" in name.lower() and "agx" not in name.lower():
                name = name.replace("Orin", "Jetson AGX Orin").replace("orin", "Jetson AGX Orin")
                
            if name:
                _GPU_NAME_CACHE = name
                return _GPU_NAME_CACHE
    except Exception:
        pass
    return ""


def get_active_device_name(use_gpu: bool = None) -> str:
    """
    Get the name of the active compute device with explicit prefix ('GPU: ...' or 'CPU: ...').
    If GPU is available and used, return 'GPU: <gpu_name>'.
    If CPU is used or CUDA is unavailable, return 'CPU: <cpu_name>'.
    """
    try:
        import torch
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        if use_gpu and torch.cuda.is_available():
            gpu_name = get_gpu_name()
            if gpu_name:
                return f"GPU: {gpu_name}"
    except Exception:
        pass
    return f"CPU: {get_cpu_name()}"



