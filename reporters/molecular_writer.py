import numpy as np

from reporters import NoReporter
from reporters.writer import Writer

class MolecularWriter(Writer):
    def __init__(self, filename, reporter=NoReporter()): #reporter):
        super().__init__(reporter)
        self.filename = filename
        self.file = open(filename, 'a')
        

    def write(self, dones:np.array, infos, rewards:np.array):
        dones = dones.reshape(-1)
        rewards = rewards.reshape(-1)
        ep_rew_final_stat = []
        for done, info, reward in zip(dones, infos, rewards):
            if done:
                str = ''.join(['{},']*(len(info.values())+1))[:-1]+'\n'
                self.file.write(str.format(*list(info.values()),reward))
                ep_rew_final_stat.append(info['final_stat'])
        self.reporter.scalar("ep_rew_final_stat", np.mean(ep_rew_final_stat))
        self.file.flush()

