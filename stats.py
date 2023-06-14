#!/usr/bin/env python
from __future__ import print_function

from datetime import datetime
import psutil
import subprocess
import sys
import time
from subprocess import check_output
import pytz
import threading
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter

#system stats sampling rate in minutes
STATS_SAMPLING_RATE = 5

class SystemStats():
    def __init__(
        self,
        tb_writer = None,
        log_dir=None,
    ): 

        self.check_nvidia_smi = self.check_nvidia_smi()

        self.tb_writer = SummaryWriter(log_dir)

        # self.gpu_queries = [
        #         'utilization.gpu',
        #         'utilization.memory',
        #         'temperature.gpu',
        #     ]
        # self.gpu_query = ','.join(self.gpu_queries)

        # Only define a thread,but haven't start it 
        self._thread = threading.Thread(target=self._thread_body) 
        self._thread.daemon = True
        
        self._shutdown = False

        self.gpu_names_counts = self.get_gpu_names_counts()


    def check_nvidia_smi(self):
        try:
            subprocess.check_output(['nvidia-smi'])
        except FileNotFoundError:
            raise EnvironmentError('The nvidia-smi command could not be found.')
    
    def get_gpu_names_counts(self):
        gpu_info = []
        gpu_count = 0
        res = subprocess.check_output(['nvidia-smi', '-L'])
        for i_res in res.decode().split('\n'):
            if i_res != '':
                gpu_info.append(i_res)
                gpu_count = gpu_count + 1
        return gpu_info, gpu_count
        # return [i_res for i_res in res.decode().split('\n') if i_res != '']
        # the output is like:
        # ['GPU 0: Tesla V100-PCIE-32GB (UUID: GPU-29ff212b-6ddd-2c59-7bb4-1022fb41b567)']

    def start(self):
        self._thread.start()

    def _thread_body(self):
        print("This is about system stats thread, number is %s" %threading.current_thread())

        start_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        end_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        time_difference = end_time - start_time # Record time interval

        second = self.sleep_time(0, 0, STATS_SAMPLING_RATE)
        tb_writer = self.tb_writer

        while True:
            stats = self.fetch_gpu_usage(time_difference)
            tb_writer.add_scalar('System/GPU Memory Allocated (%)', stats["gpu_utilization"], stats["time_period"])
            tb_writer.add_scalar('System/GPU Utilization (%)', stats["gpu_vram_usage"], stats["time_period"])
            tb_writer.add_scalar('System/temperature (C)', stats["gpu_temperature"], stats["time_period"])
            time_difference = time_difference + STATS_SAMPLING_RATE # 记录距第一次采样的每次采样的时间间隔
            time.sleep(second) # 休眠时间，单位：秒, 每次休眠5秒

            if self._shutdown:
                    break

        tb_writer.close()

    
    def shutdown(self):
        self._shutdown = True
        try:
            self._thread.join()
        # Incase we never start it
        except RuntimeError:
            pass


    def fetch_gpu_usage(self, time_difference):
        stats = {}
        
        smi_output = check_output(['nvidia-smi', '-q', '-x']).decode() 
        root = ET.fromstring(smi_output) 
        
        num_gpu     = 0
        utilization = -1
        vram_usage  = -1
        for gpu in root.iter('gpu'):
            
            stats["time_period"] = time_difference
            stats["gpu_utilization"] = int(gpu.find('utilization').find('gpu_util').text.split()[0])
            vram_used       = int(gpu.find('fb_memory_usage').find('used').text.split()[0])
            vram_total      = int(gpu.find('fb_memory_usage').find('total').text.split()[0])
            stats["gpu_vram_usage"] = int(vram_used * 100 * 1.0 / (vram_total * 1.0))
            # gpu_temperature: ValueError: invalid literal for int() with base 10: 'N/A'
            stats["gpu_temperature"] = int(gpu.find('temperature').find('gpu_temp').text.split()[0]) 
            num_gpu         += 1

        if num_gpu > 1:
            print("FIXME: only the last GPU stats collected!!!")

        # return time_period, utilization, vram_usage
        return stats
  
    def sleep_time(self, time_hour, time_min, time_second):
        return time_hour * 3600 + time_min * 60 + time_second

# Test
# system_logs = SystemStats(log_dir='./logs/test_threading')
# system_logs.start()