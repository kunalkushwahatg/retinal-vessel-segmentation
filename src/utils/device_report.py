import psutil
import platform
import time
from typing import Dict, List, Optional
import pynvml
import GPUtil

class HardwareMonitor:
    def __init__(self):
        self.cpu_info = self._get_cpu_info()
        self.gpu_info = self._get_gpu_info()
        self.nvml_initialized = True
        
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
        except pynvml.NVMLError:
            pass

    def __del__(self):
        if self.nvml_initialized:
            pynvml.nvmlShutdown()

    def _get_cpu_info(self) -> Dict:
        """Get static CPU information"""
        return {
            'name': platform.processor(),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'architecture': platform.architecture()[0],
            'system': platform.system(),
            'machine': platform.machine()
        }

    def _get_gpu_info(self) -> List[Dict]:
        """Get static GPU information"""
        gpus = []
        try:
            GPUs = GPUtil.getGPUs()
            for gpu in GPUs:
                gpus.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'driver': gpu.driver,
                    'memory_total': gpu.memoryTotal
                })
            
        except Exception as e:
            pass
        return gpus

    def get_cpu_stats(self) -> Dict:
        """Get dynamic CPU statistics"""
        try:
            temps = psutil.sensors_temperatures()
            cpu_temp = temps['coretemp'][0].current if 'coretemp' in temps else None
        except:
            cpu_temp = None

        return {
            'usage_percent': psutil.cpu_percent(interval=1),
            'memory_used': psutil.virtual_memory().used,
            'memory_total': psutil.virtual_memory().total,
            'temperature': cpu_temp,
            'frequency_current': psutil.cpu_freq().current,
            'load_avg': [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        }

    def get_gpu_stats(self) -> List[Dict]:
        """Get dynamic GPU statistics"""
        stats = []
        if not self.nvml_initialized:
            return stats

        try:
            GPUs = GPUtil.getGPUs()
            for gpu in GPUs:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                stats.append({
                    'id': gpu.id,
                    'memory_used': mem.used,
                    'memory_total': mem.total,
                    'utilization_gpu': util.gpu,
                    'utilization_mem': util.memory,
                    'temperature': temp,
                    'power_usage': pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    'fan_speed': pynvml.nvmlDeviceGetFanSpeed(handle)
                })
        except Exception as e:
            pass

        return stats

    def generate_report(self) -> str:
        """Generate a human-readable hardware report"""
        report = []
        
        # CPU Report
        cpu_static = self.cpu_info
        cpu_dynamic = self.get_cpu_stats()
        report.append("=== CPU ===")
        report.append(f"Name: {cpu_static['name']}")
        report.append(f"Cores: {cpu_static['cores_physical']} physical, {cpu_static['cores_logical']} logical")
        report.append(f"Usage: {cpu_dynamic['usage_percent']}%")
        report.append(f"Temperature: {cpu_dynamic['temperature'] or 'N/A'}°C")
        report.append(f"Memory: {self._bytes_to_gb(cpu_dynamic['memory_used'])}/{self._bytes_to_gb(cpu_dynamic['memory_total'])} GB")

        # GPU Report
        if self.gpu_info:
            gpu_dynamic = self.get_gpu_stats()
            
            for idx, (gpu_static, gpu_stat) in enumerate(zip(self.gpu_info, gpu_dynamic)):
                report.append(f"\n=== GPU {idx} ===")
                report.append(f"Name: {gpu_static['name']}")
                report.append(f"Driver: {gpu_static['driver']}")
                report.append(f"Utilization: {gpu_stat['utilization_gpu']}% GPU, {gpu_stat['utilization_mem']}% MEM")
                report.append(f"Temperature: {gpu_stat['temperature']}°C")
                report.append(f"Memory: {self._bytes_to_gb(gpu_stat['memory_used'])}/{self._bytes_to_gb(gpu_static['memory_total'])} GB")
                report.append(f"Power: {gpu_stat['power_usage']:.1f}W")
                report.append(f"Fan Speed: {gpu_stat['fan_speed']}%")

        return "\n".join(report)

    def _bytes_to_gb(self, bytes: int) -> float:
        return round(bytes / (1024**3), 2)

    def continuous_monitoring(self, interval: int = 5):
        """Continuous monitoring with specified interval (seconds)"""
        try:
            while True:
                print("\033c", end="")  # Clear console
                print(self.generate_report())
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Monitoring stopped")

# Usage example
if __name__ == "__main__":
    monitor = HardwareMonitor()
    print(monitor.generate_report())
    
    # For continuous monitoring:
    # monitor.continuous_monitoring(interval=2)