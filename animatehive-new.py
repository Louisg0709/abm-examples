#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/introduction-to-mesa-agent-based-modeling-in-python-bcb0596e1c9a
# 
# https://mesa.readthedocs.io/en/latest/tutorials/intro_tutorial.html
# 
# https://notebook.community/projectmesa/mesa-examples/examples/ForestFire/.ipynb_checkpoints/Forest%20Fire%20Model-checkpoint
# 
# To install Mesa, simply:
# 
# $ pip install mesa https://mesa.readthedocs.io/en/latest/apis/space.html

# In[337]:

    

import random

import numpy as np
import math

from hiveclasses import *

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.animation import FuncAnimation

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation



def draw_boids(model):
    x_vals = []
    y_vals = []
    t_vals=[]
    cols = []
    
    for boid in model.schedule.agents:
        x, y = boid.pos
        if boid.atype == 0:
            atype=1
            cols.append(1+atype)
        if boid.atype == 1:
            cols.append(2+boid.pollen)
            atype=10
        if boid.atype == 2:
            cols.append(2+boid.honey)
            atype=50
        #cols.append(1)
        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(100*atype)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(x_vals, y_vals, s=t_vals, c=cols, alpha=0.5)
    plt.show()


# In[343]:

#np,nb,nq,width,height
model = HiveModel(50, 500, 3, 100, 100, speed=1, vision=2, pseparation=4, bseparation=1, qseparation=10)

model.step


x_vals = []
y_vals = []
t_vals=[]
cols = []
    
for boid in model.schedule.agents:
    x, y = boid.pos
    if boid.atype == 0:
        atype=1
        cols.append(1+atype)
    if boid.atype == 1:
        cols.append(2+boid.pollen)
        atype=10
    if boid.atype == 2:
        cols.append(2+boid.honey)
        atype=50
    #cols.append(1)
    x_vals.append(x)
    y_vals.append(y)
    t_vals.append(100*atype)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
scatter=ax.scatter(x_vals, y_vals, s=t_vals, c=cols, alpha=0.5)
#plt.show()


def update(frame_number):
    model.step()
    colsp=np.array([])
    xp, yp = np.array([]), np.array([])
    for boid in model.schedule.agents:
        x, y = boid.pos
        if boid.atype == 0:
            atype=1
            ncols=(1+atype)
        if boid.atype == 1:
            ncols=(2+boid.pollen)
            atype=10
        if boid.atype == 2:
            ncols=(2+boid.honey)
            atype=50
            #cols.append(1)

        #x_vals.append(x)
        #y_vals.append(y)
        #t_vals.append(100*atype)
        xp = np.append(xp, x)
        yp = np.append(yp, y)
        colsp=np.append(colsp,ncols)
    
    #upos=np.concatenate(xp,yp,1)
    scatter.set_offsets(np.transpose([xp,yp]))
    scatter.set_array(colsp)
    print(frame_number)
    
    
    
    
    
    return scatter

print("start animation")    
anim = FuncAnimation(fig, update, interval=10)
plt.show()    
    
    
    
# In[375]:


#for i in range(10):
#    model.step()
#    print(model.time)
    #draw_boids(model)
    #input(a)



