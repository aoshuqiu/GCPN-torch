import os
import sys
import subprocess


class cd:
    """上下文管理器，用于改变当前的工作目录"""
    
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

def GCMCPredictor(cifname):
    gcmc_path = '/home/zhangjinhang/GCPN-torch/GCMC'
    with cd(gcmc_path):
        path = gcmc_path + '/gcmc.py '
        cmd = "python " + path + '\'' + cifname + '\''
        print(cmd)
        subprocess.run(cmd, shell=True)
        path = gcmc_path + '/ans.txt'
        with open(path, 'r') as f:
            for line in f:
                str = line
    return str


