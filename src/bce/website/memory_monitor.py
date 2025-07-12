#!/usr/bin/env python3
"""
Memory monitoring script for BCE Prediction Web Server
用于监控BCE预测过程中的内存使用情况
"""

import os
import sys
import time
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import subprocess
import signal

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, log_file: str = None, interval: float = 1.0):
        self.interval = interval
        self.log_file = log_file or f"memory_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.monitoring = False
        self.thread = None
        self.data = []
        self.start_time = None
        self.server_pid = None
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"内存监控已启动，日志文件：{self.log_file}")
        
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.thread:
            self.thread.join()
            
        self._save_data()
        print(f"内存监控已停止，数据已保存到：{self.log_file}")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                data_point = self._collect_data()
                self.data.append(data_point)
                time.sleep(self.interval)
            except Exception as e:
                print(f"监控过程中出错：{e}")
                
    def _collect_data(self) -> Dict:
        """收集内存数据"""
        # 系统内存信息
        system_memory = psutil.virtual_memory()
        
        # 当前进程信息
        current_process = psutil.Process()
        current_memory = current_process.memory_info()
        
        # 查找BCE服务器进程
        server_memory = None
        if self.server_pid:
            try:
                server_process = psutil.Process(self.server_pid)
                server_memory = server_process.memory_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                server_memory = None
        
        # 查找所有Python进程
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline'] or []
                    if any('bce' in arg.lower() or 'main.py' in arg for arg in cmdline):
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'cmdline': ' '.join(cmdline)
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return {
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
            'system_memory': {
                'total_mb': system_memory.total / 1024 / 1024,
                'available_mb': system_memory.available / 1024 / 1024,
                'used_mb': system_memory.used / 1024 / 1024,
                'used_percent': system_memory.percent
            },
            'current_process': {
                'pid': current_process.pid,
                'rss_mb': current_memory.rss / 1024 / 1024,
                'vms_mb': current_memory.vms / 1024 / 1024
            },
            'server_process': {
                'pid': self.server_pid,
                'rss_mb': server_memory.rss / 1024 / 1024 if server_memory else 0,
                'vms_mb': server_memory.vms / 1024 / 1024 if server_memory else 0
            } if server_memory else None,
            'python_processes': python_processes
        }
        
    def _save_data(self):
        """保存数据到文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            
    def get_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.data:
            return {}
            
        system_memory_used = [d['system_memory']['used_mb'] for d in self.data]
        system_memory_percent = [d['system_memory']['used_percent'] for d in self.data]
        current_process_rss = [d['current_process']['rss_mb'] for d in self.data]
        
        # 服务器进程内存（如果有）
        server_process_rss = []
        for d in self.data:
            if d.get('server_process'):
                server_process_rss.append(d['server_process']['rss_mb'])
        
        # 所有Python进程总内存
        total_python_memory = []
        for d in self.data:
            total_mem = sum(p['memory_mb'] for p in d['python_processes'])
            total_python_memory.append(total_mem)
        
        return {
            'monitoring_duration': self.data[-1]['elapsed_time'] if self.data else 0,
            'data_points': len(self.data),
            'system_memory': {
                'peak_used_mb': max(system_memory_used) if system_memory_used else 0,
                'peak_used_percent': max(system_memory_percent) if system_memory_percent else 0,
                'avg_used_mb': sum(system_memory_used) / len(system_memory_used) if system_memory_used else 0,
                'avg_used_percent': sum(system_memory_percent) / len(system_memory_percent) if system_memory_percent else 0
            },
            'current_process': {
                'peak_rss_mb': max(current_process_rss) if current_process_rss else 0,
                'avg_rss_mb': sum(current_process_rss) / len(current_process_rss) if current_process_rss else 0
            },
            'server_process': {
                'peak_rss_mb': max(server_process_rss) if server_process_rss else 0,
                'avg_rss_mb': sum(server_process_rss) / len(server_process_rss) if server_process_rss else 0
            } if server_process_rss else None,
            'total_python_processes': {
                'peak_total_mb': max(total_python_memory) if total_python_memory else 0,
                'avg_total_mb': sum(total_python_memory) / len(total_python_memory) if total_python_memory else 0
            }
        }
        
    def set_server_pid(self, pid: int):
        """设置服务器进程PID"""
        self.server_pid = pid
        
    def print_summary(self):
        """打印内存使用摘要"""
        summary = self.get_summary()
        if not summary:
            print("没有收集到内存数据")
            return
            
        print("\n" + "="*60)
        print("内存使用摘要报告")
        print("="*60)
        print(f"监控时长: {summary['monitoring_duration']:.1f} 秒")
        print(f"数据点数: {summary['data_points']}")
        print()
        
        print("系统内存使用:")
        sys_mem = summary['system_memory']
        print(f"  峰值使用: {sys_mem['peak_used_mb']:.1f} MB ({sys_mem['peak_used_percent']:.1f}%)")
        print(f"  平均使用: {sys_mem['avg_used_mb']:.1f} MB ({sys_mem['avg_used_percent']:.1f}%)")
        print()
        
        print("当前进程内存使用:")
        curr_proc = summary['current_process']
        print(f"  峰值RSS: {curr_proc['peak_rss_mb']:.1f} MB")
        print(f"  平均RSS: {curr_proc['avg_rss_mb']:.1f} MB")
        print()
        
        if summary.get('server_process'):
            print("服务器进程内存使用:")
            server_proc = summary['server_process']
            print(f"  峰值RSS: {server_proc['peak_rss_mb']:.1f} MB")
            print(f"  平均RSS: {server_proc['avg_rss_mb']:.1f} MB")
            print()
        
        print("所有Python进程总内存使用:")
        python_total = summary['total_python_processes']
        print(f"  峰值总计: {python_total['peak_total_mb']:.1f} MB")
        print(f"  平均总计: {python_total['avg_total_mb']:.1f} MB")
        print()
        
        print(f"详细数据已保存到: {self.log_file}")
        print("="*60)


class ServerMemoryTester:
    """服务器内存测试器"""
    
    def __init__(self, server_script: str = "run_server.py", port: int = 8000):
        self.server_script = server_script
        self.port = port
        self.server_process = None
        self.monitor = MemoryMonitor()
        
    def start_server(self):
        """启动服务器"""
        script_path = Path(__file__).parent / self.server_script
        if not script_path.exists():
            raise FileNotFoundError(f"服务器脚本不存在: {script_path}")
            
        # 启动服务器进程
        cmd = [sys.executable, str(script_path), "--port", str(self.port), "--preload"]
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务器启动
        time.sleep(3)
        
        # 检查服务器是否正在运行
        if self.server_process.poll() is not None:
            stdout, stderr = self.server_process.communicate()
            raise RuntimeError(f"服务器启动失败:\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")
            
        print(f"服务器已启动，PID: {self.server_process.pid}")
        self.monitor.set_server_pid(self.server_process.pid)
        
    def stop_server(self):
        """停止服务器"""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            print("服务器已停止")
            
    def run_test(self, test_duration: int = 300):
        """运行测试"""
        print(f"开始内存测试，测试时长: {test_duration} 秒")
        
        try:
            # 启动服务器
            self.start_server()
            
            # 开始监控
            self.monitor.start_monitoring()
            
            # 等待测试完成
            print(f"测试正在进行中...")
            print(f"请在浏览器中访问 http://localhost:{self.port} 进行预测测试")
            print("按 Ctrl+C 可以提前结束测试")
            
            try:
                time.sleep(test_duration)
            except KeyboardInterrupt:
                print("\n用户中断测试")
                
        finally:
            # 停止监控
            self.monitor.stop_monitoring()
            
            # 停止服务器
            self.stop_server()
            
            # 打印摘要
            self.monitor.print_summary()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BCE预测服务器内存监控工具')
    parser.add_argument('--mode', choices=['monitor', 'test'], default='monitor',
                        help='运行模式: monitor(仅监控) 或 test(启动服务器并测试)')
    parser.add_argument('--duration', type=int, default=300,
                        help='测试持续时间（秒），默认300秒')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='监控间隔（秒），默认1秒')
    parser.add_argument('--port', type=int, default=8000,
                        help='服务器端口，默认8000')
    parser.add_argument('--log-file', type=str,
                        help='日志文件名，默认使用时间戳')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        # 仅监控模式
        monitor = MemoryMonitor(args.log_file, args.interval)
        
        def signal_handler(signum, frame):
            print("\n收到中断信号，正在停止监控...")
            monitor.stop_monitoring()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        monitor.start_monitoring()
        
        try:
            print("内存监控已启动，按 Ctrl+C 停止")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stop_monitoring()
            monitor.print_summary()
            
    elif args.mode == 'test':
        # 测试模式
        tester = ServerMemoryTester(port=args.port)
        tester.run_test(args.duration)


if __name__ == "__main__":
    main() 