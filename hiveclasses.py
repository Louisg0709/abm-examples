# -*- coding: utf-8 -*-
"""
Hive Classes

Created on Sun Jul  3 12:36:05 2022
@author: louis
"""

import random

import numpy as np
import math

import matplotlib.pyplot as plt
#%matplotlib inline  #for the jupyter notebook

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation


#from mesa.datacollection import DataCollector
#from mesa.batchrunner import BatchRunner


class HiveModel(Model):
    '''
    Hive model class. Handles agent creation, placement and scheduling.
    '''

    def __init__(self, NP, NB, NQ, width, height, speed, vision, pseparation, bseparation, qseparation):
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
        self.NP = NP
        self.NB = NB
        self.NQ = NQ
        
        self.vision = vision
        self.speed = speed
        self.qseparation = qseparation
        self.pseparation = pseparation
        self.bseparation = bseparation
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, True, 10, 10)
        self.make_agents()
        self.running = True
        self.time =0

    def make_agents(self):
        '''
        Create N agents, with random positions and starting headings.
        '''
        for i in range(self.NB):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = (x, y)
            ang=2*math.pi*np.random.random(1)
            heading=np.array((math.cos(ang),math.sin(ang)))
            #heading = np.random.random(2) * 2 - np.array((1, 1))
            #heading /= np.linalg.norm(heading)
            bee = Bee(i, self, pos, 0, 0,5,self.speed, heading, self.vision,self.bseparation)
            self.space.place_agent(bee, pos)
            self.schedule.add(bee)
        for i in range(self.NP):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = (x, y)
            plant = Plant(i+self.NB, self, pos, 5, self.vision,
                        self.pseparation)
            self.space.place_agent(plant, pos)
            self.schedule.add(plant)
        for i in range(self.NQ):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = (x, y)
            queen = Queen(i+self.NB+self.NP, self, pos, 5, 5, self.vision,
                        self.qseparation)
            self.space.place_agent(queen, pos)
            self.schedule.add(queen)            


    def step(self):
        self.time = self.time+1
        self.schedule.step()

class Bee(Agent):
    '''
    A Bee Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring  queen agents.
        - Separation: avoiding getting too close to any other worker agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Bees have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and heading (a unit vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, pollen=0, food=5, health=5, speed=0.1, heading=None,
                 vision=5, separation=1, atype=0):
        '''
        Create a new Bee flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Queens/Plants.
            separation: Minimum distance to maintain from other Boids.
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.food = food
        self.health = health
        self.pollen = pollen
        self.speed = speed
        self.atype=atype
        self.honey=0
        self.randomstateflag = 0
        self.randomstatecycle =6
        self.randomstatestep = 0
        self.model=model
        if heading is not None:
            self.heading = heading
        else:
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))
            #self.heading = 2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)
            #self.heading /= np.linalg.norm(self.heading)

        self.vision = vision
        self.separation = separation

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''
        center = np.array([0.0, 0.0])
        count=0
        for neighbor in neighbors:
            if neighbor.atype == 0:
                center += np.array(neighbor.pos)
                count = count +1
                
        if count>0: 
            heading=center / count
        else:
            ang=2*math.pi*np.random.random(1)
            heading=np.array((math.cos(ang),math.sin(ang)))
            #heading=2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)            
            #heading/=np.linalg.norm(heading)
        return np.float64(heading)
    
    def cohereplant(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''

        center = np.array([0.0, 0.0])
        count=0
        nsep=9e99
        nhead=np.random.random(2)
        for neighbor in neighbors:
            if neighbor.atype == 1:
                sepv = np.float64(np.array(neighbor.pos)-self.pos)
                sep=np.linalg.norm(sepv)
                if sep<nsep:
                    nsep=sep
                    nhead=sepv
                count = count +1
 
        if count<=0: 
            ang=2*math.pi*np.random.random(1)
            nhead=np.array((math.cos(ang),math.sin(ang)))
            #nhead=2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)            
            #nhead/=np.linalg.norm(nhead)
        return nhead
          
        

    def coherequeen(self, neighbors):
        '''
        Return the vector toward the locationn of the nearest queen.
        '''

        center = np.array([0.0, 0.0])
        count=0
        nsep=9e99
        nhead=np.random.random(2)
        for neighbor in neighbors:
            if neighbor.atype == 2:
                sepv = np.float64(np.array(neighbor.pos)-self.pos)
                sep=np.linalg.norm(sepv)
                if sep<nsep:
                    nsep=sep
                    nhead=sepv
                count = count +1
                
        if count<=0: 
            #nhead=2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)            
            #nhead/=np.linalg.norm(nhead)
            ang=2*math.pi*np.random.random(1)
            nhead=np.array((math.cos(ang),math.sin(ang)))

        return nhead
       
        

    
    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist. Only do this if not another bee
        and if not near a plant or a queen
        '''
        my_pos = np.float64(np.array(self.pos))
        sep_vector = np.float64(np.array([0, 0]))
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.float64(np.array(neighbor.pos))
                dist = np.linalg.norm(my_pos - their_pos)
                if dist < self.separation:
                    sep_vector -= np.float64(their_pos - my_pos)
        return np.float64(sep_vector)


    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''

        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)

        
        if len(neighbors) > 0:
            cohere_vector = np.float64(self.cohere(neighbors))
            separate_vector = np.float64(self.separate(neighbors))
            self.heading += np.float64((cohere_vector +
                             separate_vector ))
            self.heading = np.float64(self.heading)/np.int64(np.linalg.norm(self.heading))
        
        
        
        #need to check the heading very carefully as this is resulting in nan values!!@!!!!!!            
        if len(neighbors) > 0:
            if self.honey < 2 and self.pollen>1:
                self.heading=self.coherequeen(neighbors)
            if self.pollen <= 1 and self.honey>2:
                self.heading=self.cohereplant(neighbors)


            
   
        #put the bee into a random search
        if np.random.random(1)>0.8 and self.randomstateflag == 0:
            self.randomstateflag=1
            self.randomstatestep=0
            
           
        if self.randomstateflag == 1:
            self.randomstatestep += 1
            if self.randomstatestep>self.randomstatecycle:
                self.randomstateflag=0
                self.randomstatestep=0
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))
            #heading=np.random.random(2)
            #heading/=np.linalg.norm(heading)
            #self.heading=heading
        
        #heading=np.random.random(2)
        #center = np.array([-1, -1])
        #self.heading+=np.int64(center)
        #heading/=np.linalg.norm(heading)
        #self.heading=heading
        #self.speed=0.1

        #ang=2*math.pi*np.random.random(1)
        #self.heading=np.array((math.cos(ang),math.sin(ang)))

        

            
        
        new_pos = np.array(self.pos) + self.heading * self.speed
        new_x, new_y = new_pos
        if math.isnan(new_x) or math.isnan(new_y):
            print("nan pos detected ")
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))
            #self.heading = np.random.random(2)
            new_pos = np.array(self.pos) + self.heading * self.speed
            new_x, new_y = new_pos
        else:
            if new_x>self.model.space.x_max:
                new_x=self.model.space.x_min
            if new_x<self.model.space.x_min:
                new_x=self.model.space.x_max
            if new_y>self.model.space.y_max:
                new_y=self.model.space.y_min
            if new_y<self.model.space.y_min:
                new_y=self.model.space.y_max
            self.model.space.move_agent(self, (new_x, new_y))
            
class Plant(Agent):
    '''
    A Plant agent.

    Plant produes pollen bees take pollen from plant

    Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, pollen =5, vision = 2, separation=1, atype=1):
        '''
        Create a new Plant agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            separation: Minimum distance to maintain from other Boids to give pollen.
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.separation = separation
        self.pollen = 5
        self.atype = atype
        self.vision = vision
        self.time = 0


    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''

        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        time=self.model.time
        my_pos = np.array(self.pos)
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    #print("pollen transfer", self.pollen, neighbor.pollen)
                    if self.pollen > 0:
                        neighbor.pollen = neighbor.pollen + 2
                        self.pollen = self.pollen - 2
                        if self.pollen > 0 and neighbor.pollen<5:
                            neighbor.speed=0.05
                        else:
                            neighbor.speed=0.5
                        #print ('Pollen transfered', self.pollen, neighbor.pollen)
        
        if self.model.time % 30 == 0:
            #print("hooray more pollen")
            self.pollen=self.pollen+random.random()/20
            
class Queen(Agent):
    '''
    A Queen agent.

    Uses pollen to make food
    get pollen from bee
    queen make food
    queen give food to bee

    Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, pollen=5, honey=5, vision=5, separation=1, atype=2):
        '''
        Create a new Queen agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Queens/Plants.
            separation: Minimum distance to maintain from other Boids.
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.separation = separation
        self.pollen = pollen
        self.honey = honey
        self.atype = atype
        self.vision = vision

    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''

        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        nworkers=0
        health=0
        my_pos = np.array(self.pos)
        #self.separation = 10
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                if dist < self.separation:
                    nworkers=nworkers+1
                    health=health+neighbor.health
                    if neighbor.pollen > 0:
                        neighbor.pollen = neighbor.pollen - 1
                        self.pollen = self.pollen + 1
                if dist < self.separation:
                    print("Honey before",self.honey, neighbor.honey)
                    if self.honey > 0:
                        neighbor.honey = neighbor.honey + 1
                        self.honey = self.honey - 1
                        print("honey given", self.honey, neighbor.honey, self.unique_id)
                        if neighbor.pollen>0:
                            neighbor.speed = 0
                        else: 
                            neighbor.speed = 0.1
    
        #calculate number of workers near and use to make honey
        if nworkers>0:
            if self.pollen>0:
                self.honey=self.honey+self.pollen*(health/nworkers)
                self.pollen=self.pollen-nworkers
                

