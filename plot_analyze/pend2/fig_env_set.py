# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 4:36 PM
@desc:
"""
from matplotlib import pyplot as plt


plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"]  = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

wordsize = 18
plt.rc('font', size=wordsize)  # controls default text sizes
plt.rc('axes', titlesize=wordsize)  # fontsize of the axes title
plt.rc('axes', labelsize=wordsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=wordsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=wordsize)  # fontsize of the tick labels
plt.rc('legend', fontsize=wordsize)  # legend fontsize
plt.rc('figure', titlesize=wordsize)  # fontsize of the figure title

DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 30  # 100 for pend-sim, 30 for pend-real
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
RK4 = ''

result_dir='./result'