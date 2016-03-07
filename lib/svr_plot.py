#-*- coding:utf-8 -*-
# basic
import os
import copy
import json                       # json encoder/decoder
import operator                   # for sorting dict
import timeit                     # for benchmark
import random 
import math
import numpy as np
from operator import itemgetter   # for sorting list of list
from datetime import datetime   

# plot
import matplotlib
import matplotlib.pyplot as plt   # plot plate
import mpld3                      # plot to web
# import matplotlib.cm as cm
from matplotlib import cm
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/usr/share/fonts/truetype/LiHeiPro.ttf", size=11)   # chinese font for matplotlib
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import sem

# marsan
from lib import adhub_enum as num

#================================
#  plot performance
#================================
def plot_predict(Y, Yp, X=None, sort=True, sort_both=False):
  # X-axis
  if (X == None): X = np.arange(len(Y))
  
  # sort for clear plot
  if sort:
    tmp = []
    for i in np.arange(len(Y)): 
      tmp.append([Y[i], Yp[i]])
    tmp = sorted(tmp, key=itemgetter(0))
    ltmp = len(tmp)
    Y_s = np.zeros(ltmp)
    Yp_s = np.zeros(ltmp)
    for i in np.arange(ltmp):
      Y_s[i] = tmp[i][0]
      Yp_s[i] = tmp[i][1]
  else:
    Y_s = Y
    Yp_s = Yp

  # data & predict
  fig, axes = plt.subplots(1, 3, figsize=(18, 4))
  axes[0].plot(X, Yp_s, c="g", label="predict", linewidth=1)
  axes[0].plot(X, Y_s, c="k", label="data", linewidth=1)
  axes[0].set_xlabel("num")
  axes[0].set_ylabel("val")
  axes[0].set_title("SVR - data & predict")
  # plt_s1.legend()

  if sort_both:
    Ye = Yp_s - Y_s
    Y_s = sorted(Y)
    Yp_s = sorted(Yp)
    axes[1].plot(X, Yp_s, c="g", label="predict", linewidth=1)
    axes[1].plot(X, Y_s, c="k", label="data", linewidth=1)
    axes[1].set_xlabel("num")
    axes[1].set_ylabel("val")
    axes[1].set_title("SVR - data & predict")
  else:
    # error
    Ye = Yp_s - Y_s
    axes[1].plot(X, Ye, c="r", label="error", linewidth=1)
    axes[1].set_xlabel("num")
    axes[1].set_ylabel("val")
    axes[1].set_title("SVR - error")

  # sorted error
  Ye_s = sorted(Ye)    
  axes[2].plot(X, Ye_s, c="r", label="error_sorted", linewidth=1)
  axes[2].set_title("SVR - error sorted")
  plt.show()

def plot_grid_3d(X_range, Y_range, Z, X_label='X', Y_label='Y', Z_label='Z', json_data=False):
  fig = plt.figure(figsize=(18,6))
  ax = fig.add_subplot(1, 1, 1, projection='3d')

  X, Y = np.meshgrid(X_range, Y_range)
  Zm = Z #zip(*Z)
  # print "[plot_check]", np.shape(X_range), np.shape(Y_range), np.shape(X), np.shape(Y), np.shape(Z)
  p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  x_offset = (max(X_range) - min(X_range))*0.2
  y_offset = (max(Y_range) - min(Y_range))*0.2
  Zmax = max(max(Zm))
  Zmin = min(min(Zm))
  z_offset = (Zmax - Zmin)*0.2
  cset = ax.contour(X, Y, Z, zdir='x', offset=X_range[0]-x_offset, cmap=cm.coolwarm)
  cset = ax.contour(X, Y, Z, zdir='y', offset=Y_range[-1]+y_offset, cmap=cm.coolwarm)
  cset = ax.contour(X, Y, Z, zdir='z', offset=Zmin-z_offset, cmap=cm.coolwarm)
  ax.set_xlabel(X_label)
  ax.set_ylabel(Y_label)
  ax.set_zlabel(Z_label)
  ax.set_xlim(X_range[0]-x_offset, X_range[-1])
  ax.set_ylim(Y_range[0], Y_range[-1]+y_offset)
  ax.set_zlim(Zmin-z_offset, Zmax+z_offset)
  cb = fig.colorbar(p, shrink=0.5)
  # print "[mpld3] before json serialize."
  if json_data: return mpld3.fig_to_dict(fig)

def plot_grid_2d(X, Z, X_label='X', Z_label='Z', json_data=False):
  fig = plt.figure(figsize=(6,2.5))
  ax = fig.add_subplot(1, 1, 1)
  p = ax.plot(X, Z)
  ax.set_xlabel(X_label)
  ax.set_ylabel(Z_label)
  if json_data:
    chart_data = mpld3.fig_to_dict(fig)
    plt.close()
    return chart_data

def plot_svr_grid(gs_svr):
  # extract X,Y
  sample_x = []
  sample_y = []
  for s in gs_svr.grid_scores_:
    g = math.log(s.parameters['gamma'])
    c = math.log(s.parameters['C'])
    if not g in sample_x: sample_x.append(g) 
    if not c in sample_y: sample_y.append(c) 
  # print sample_x, sample_y
  X,Y = np.meshgrid(sample_x, sample_y)
  # extract Z
  Z = []
  tmp = []
  for idx, g in enumerate(gs_svr.grid_scores_):
    tmp.append(g.mean_validation_score)
    if (idx % len(sample_y) == (len(sample_y)-1)):
      Z.append(tmp)
      tmp = []
  plot_grid_3d(sample_x, sample_y, Z, X_label='gamma', Y_label='C', Z_label='R^2')

#--------------------------------
#   VIP verticals results
#--------------------------------
def plot_r2_nos(axes, X, Y_r2, Y_train, chart_num, title, y_label):
  print "\n=====[%s-%s]================================" % (y_label, title)
  print " / ".join([("%i: %s" % (idx, x)) for idx, x in enumerate(X)])
  axes[chart_num].bar(range(len(Y_r2)), Y_r2, align='center', color="y")
  axes[chart_num].set_ylabel("R2")
  axes[chart_num].set_xlabel(y_label)
  axes[chart_num].set_title(title)

  axes2 = axes[chart_num].twinx()
  axes2.plot(range(len(Y_train)), Y_train, color="r", linestyle='-', marker='o', linewidth=2.0)
  axes2.set_ylabel("samples")

def plot_r2_esb(axes, X, Ycnt, Y1, Y2, Y3, chart_num, title, y_label):
  print "\n=====[%s-%s]================================" % (y_label, title)
  print " / ".join([("%i: %s" % (idx, x)) for idx, x in enumerate(X)])
  # print Y1, Y2, Y3
  w = 0.2
  X1 = map(lambda x: x-w, range(len(Y1)))
  X2 = map(lambda x: x,   range(len(Y2)))
  X3 = map(lambda x: x+w, range(len(Y3)))
  axes[chart_num].bar(X1, Y1, width=w, align='center', color="r")
  axes[chart_num].bar(X2, Y2, width=w, align='center', color="g")
  axes[chart_num].bar(X3, Y3, width=w, align='center', color="y")
  axes[chart_num].set_ylabel("R2")
  axes[chart_num].set_xlabel(y_label)
  axes[chart_num].set_title(title)

  axes2 = axes[chart_num].twinx()
  axes2.plot(range(len(Ycnt)), Ycnt, color="k", linestyle='-', marker='o', linewidth=2.0)
  axes2.set_ylabel("samples")

