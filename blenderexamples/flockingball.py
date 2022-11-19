#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:03:04 2022

@author: mikegriffiths
"""
#bouncing ball

#Created on Mon Oct 10 20:17:54 2022

#@author: mikegriffiths

#https://github.com/mteretome/Blender-Scripts

import bpy
import math
import random
import numpy as np
from mesa import Agent
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

class Boid(Agent):
    '''
    A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents.
        - Separation: avoiding getting too close to any other agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and heading (a unit vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''
    def __init__(self, unique_id, model, pos, speed=5, heading=None,
                 vision=5, separation=1):
        '''
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Boids.
            separation: Minimum distance to maintain from other Boids.
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.speed = speed
        if heading is not None:
            self.heading = heading
        else:
            self.heading = np.random.random(2)
            self.heading /= np.linalg.norm(self.heading)
        self.vision = vision
        self.separation = separation

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''
        center = np.array([0.0, 0.0])
        for neighbor in neighbors:
            center += np.array(neighbor.pos)
        return center / len(neighbors)

    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist.
        '''
        my_pos = np.array(self.pos)
        sep_vector = np.array([0, 0])
        for neighbor in neighbors:
            their_pos = np.array(neighbor.pos)
            dist = np.linalg.norm(my_pos - their_pos)
            if dist < self.separation:
                sep_vector -= np.int64(their_pos - my_pos)
        return sep_vector

    def match_heading(self, neighbors):
        '''
        Return a vector of the neighbors' average heading.
        '''
        mean_heading = np.array([0, 0])
        for neighbor in neighbors:
            mean_heading += np.int64(neighbor.heading)
        return mean_heading / len(neighbors)

    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''

        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        if len(neighbors) > 0:
            cohere_vector = self.cohere(neighbors)
            separate_vector = self.separate(neighbors)
            match_heading_vector = self.match_heading(neighbors)
            self.heading += (cohere_vector +
                             separate_vector +
                             match_heading_vector)
            self.heading /= np.linalg.norm(self.heading)
        new_pos = np.array(self.pos) + self.heading * self.speed
        new_x, new_y = new_pos
        self.model.space.move_agent(self, (new_x, new_y))



class BoidModel(Model):
    '''
    Flocker model class. Handles agent creation, placement and scheduling.
    '''

    def __init__(self, N, width, height, speed, vision, separation):
        '''
        Create a new Flockers model.

        Args:
            N: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separtion: What's the minimum distance each Boid will attempt to
                       keep from any other
        '''
        
        #ContinuousSpace(x_max: float, y_max: float, torus: bool, x_min: float = 0, y_min: float = 0)
        self.N = N
        self.vision = vision
        self.speed = speed
        self.separation = separation
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, True, 10, 10)
        self.make_agents()
        self.running = True

    def make_agents(self):
        '''
        Create N agents, with random positions and starting headings.
        '''
        for i in range(self.N):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = (x, y)
            heading = np.random.random(2) * 2 - np.array((1, 1))
            heading /= np.linalg.norm(heading)
            boid = Boid(i, self, pos, self.speed, heading, self.vision,
                        self.separation)
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
        self.schedule.step()










#generate n balls in random locations & returns list with balls
def create_ball():
    #balls = []
    #makes the objects
    #for i in range(n): 
    x, y, z  = random.randint(-10,10), random.randint(-10,10), 0  
    bpy.ops.mesh.primitive_uv_sphere_add( 
    location = [ x, y, z] )
    ob = bpy.ops.object
    
    #smoothes the spheres 
    bpy.ops.object.shade_smooth()
    bpy.ops.object.shade_smooth()
    bpy.ops.object.modifier_add(type='SUBSURF')
    return ob   

      
        

#function to delete ALL objects       
def del_all():
    bpy.ops.object.select_all(action='SELECT')
    for ob in bpy.context.selectable_objects:
        bpy.ops.object.delete(use_global=False)
        
        
#bounce         
def bounce(i,frame_num):
    #bpy.ops.object.select_all(action='SELECT')
    for ob in bpy.context.selectable_objects:
        x, y = ob.location[0], ob.location[1]
        #z = math.sin(i)*10+10 #using sin to replicated bouncing
        z=random.randint(-10,10)
        ob.location = (x,y,z)
        ob.keyframe_insert(data_path="location", index = -1)
        #z = 10
    #asssigns positiosp and creates bouncing
    bpy.context.scene.frame_set(frame_num)
    #ob.location = (x,y,0)
    #ob.keyframe_insert(data_path="location", index = -1)
    frame_num +=10
    


#bounce         
def moveagent(model,i,frame_num):
    x_vals=[]
    y_vals=[]
    for boid in model.schedule.agents:
        x, y = boid.pos
        x_vals.append(x)
        y_vals.append(y)
    
    i=0
    #bpy.ops.object.select_all(action='SELECT')
    for ob in bpy.context.selectable_objects:
        #x, y = ob.location[0], ob.location[1]
        #z = math.sin(i)*10+10 #using sin to replicated bouncing
        #z=random.randint(-10,10)
        z=0
        ob.location = (x_vals[i],y_vals[i],z)
        #ob.location = (x,y,z)
        ob.keyframe_insert(data_path="location", index = -1)
        i=i+1
        #z = 10
    #asssigns positiosp and creates bouncing
    bpy.context.scene.frame_set(frame_num)
    #ob.location = (x,y,0)
    #ob.keyframe_insert(data_path="location", index = -1)
    frame_num +=10
        


nagents=30  
#first step create the agents #comnment out after running this step   
#del_all()


for i in range(0,nagents):
    print(i)
    ball = create_ball() 



#balls = create_balls(10)  
# N, width, height, speed, vision, separation 
model = BoidModel(nagents+1, 200, 200, speed=2, vision=20, separation=2)



#run the bounce ball step  
for i in  range(200):   
    #    bounce(i,i*10)
    model.step()
    moveagent(model,i,i*10)

