import subprocess
import re
import os
import sys


path = r'/home/zhangjinhang/GCPN-torch/GCMC'
arg1 = sys.argv[1]


def first_step(cif):
    #deldir()
    '''
    这是GCMC计算的第一步：
    修改simulation.input文件中‘FrameworkName’一行，
    换成本次计算的cif文件的名称
    运行run1
    '''
    print(cif)
    change_input('1',cif)
    cmd = './run'
    print('First : Start')
    result = subprocess.run(cmd, shell=True)
    print('First : Finished')
    print('---------------------------------------------\n\n')
    '''
    在‘/Output/System_0’中找到‘.data’文件
    在该文件的'Average Widom Rosenbluth-weight:'一行中，
    找到后面跟着的数
    例如：
    找出字符串
    ‘        [helium] Average Widom Rosenbluth-weight:   0.818291 +/- 0.003347 [-]’
    中的‘0.818291’
    '''
    dir_path = path + r'/Output/System_0'
    data_files = [f for f in os.listdir(dir_path) if f.endswith('.data')]
    str = data_files[0]
    file_path = dir_path + '/' + str
    target = 'Average Widom Rosenbluth-weight:'
    file = open(file_path, 'r')
    for line in file:
        if target in line:
            str = line
    file.close()
    # 使用正则表达式匹配出需要的部分
    match = re.search(r"\d+\.\d+", str)
    # 判断是否匹配成功
    if match:
        # 输出匹配到的结果
        str = match.group()
    else:
        print("No match found.")
    deldir()
    return str

def second_step(temp,cif):
    '''
        这是GCMC计算的第二步：
        修改simulation.input文件中‘HeliumVoidFraction’一行，
        换成刚才找到的值
        运行run
    '''
    target = 'HeliumVoidFraction'
    file_path = path + r'/simulation2.input'
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if target in line:
            str = 'HeliumVoidFraction            ' + temp
            lines[i] = str + '\n'

    with open('simulation.input', 'w') as f:
        f.write(''.join(lines))
    
    change_input('2',cif)
    #改完之后运行run
    cmd = './run'
    print('Second : Start')
    result = subprocess.run(cmd, shell=True)
    print('Second : Finished')
    print('---------------------------------------------\n\n')
    '''
        在‘/Output/System_0’中找到‘.data’文件
        在该文件的'Average loading absolute'一行中，
        找到后面跟着的数，也就是吸附值
    '''
    dir_path = path + r'/Output/System_0'
    data_files = [f for f in os.listdir(dir_path) if f.endswith('.data')]
    str = data_files[0]
    file_path = dir_path + '/' + str

    target = 'Average loading absolute            '
    file = open(file_path, 'r')
    for line in file:
        if target in line:
            str = line
            break
    file.close()
    # 使用正则表达式匹配出需要的部分
    match = re.search(r"\d+\.\d+", str)
    # 判断是否匹配成功
    if match:
        # 输出匹配到的结果
        str = match.group()
    else:
        print("No match found.")
    deldir()
    return str

def change_input(idx,cif):
    file_path = 'simulation' + idx + '.input'
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'FrameworkName' in line:
            str = 'FrameworkName ' + cif
            lines[i] = str + '\n'

    with open('simulation.input', 'w') as f:
        f.write(''.join(lines))

def deldir():
    #"rm - rf"   删除产生的四个文件夹
    cmd = "rm -r " + path + "/Movies"
    subprocess.run(cmd, shell=True)
    cmd = "rm -r " + path + "/Restart"
    subprocess.run(cmd, shell=True)
    cmd = "rm -r " + path + "/VTK"
    subprocess.run(cmd, shell=True)
    cmd = "rm -r " + path + "/Output"
    subprocess.run(cmd, shell=True)


cif = arg1
str = first_step(cif)
str = second_step(str,cif)
print(str)


ans = str
path = path + '/ans.txt'
with open(path, 'w') as f:
    f.write(ans)
#export RASPA_DIR=${HOME}/RASPA/simulations