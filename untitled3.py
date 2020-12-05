#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:16:02 2020

@author: marionlelievre
"""
from dataclasses import dataclass
from enum import Enum
from random import *
from math import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
sns.set()

# Some constants 

FRAME_RATE = 1          # Refresh graphics very FRAME_RATE hours 
DENSITY = 100
I0 = 0.03 
SOCIAL_DISTANCE = 0.007 # in km 
SPEED = 6               # km/day 
BETA1 = 0.50            # Probality to gets infected (From "S" to "I") 
BETA2 = 0.8               # Probability to get infected if you are part of an infected cluster
GAMMA1 = 7 * 24         # Number of hours before recovering (From "I" to "R") 
GAMMA2 = 0.003          # Probability to die (From "I" to "D") 
EPSILON = 0.05          # Probability to be Susceptible again (From "R" to "S") 
BORDER = True
SIGMA = ((6/24)/(3* sqrt(2))) # en km
LOCKDOWN= False
proba_for_asymptomatique = 0.3
proba_for_infectious = 0.05
MAX_HOME_DISTANCE = 0.1
## The locations of borders on the map


## The locations of borders on the map

A = (0.,0.82142857)
B = (0.46896552,0.53125)
C = (0.57241379,0.58928571)
D = (1.,0.33035714)


class SIRState(Enum):
    SUSCEPTIBLE = 0
    INFECTIOUS = 1
    RECOVERED = 2
    DEAD = 3

class District(Enum):
    D7 = 0
    D15 = 1

def compute_district(x,y):
    fy = my_piecewise_curve(x)
    if y < fy:
        return District(1)
    else:
        return District(0)

def my_piecewise_curve(x):   #retourne le y
    if 0<=x< B[0]:
        return ((B[1]-A[1])/B[0]-A[0])*x+A[1]
    if B[0]<=x <C[0]:
        return ((C[1]-B[1]) /(C[0]-B[0]))*x+ (B[1]-(((C[1]-B[1]) /(C[0]-B[0]))*B[0]))
    if C[0]<= x:
        return ((D[1]-C[1])/(D[0]-C[0]))*x+ (C[1]-((D[1]-C[1])/(D[0]-C[0]))*C[0])
    
@dataclass
class Person:
    x: float    #Normalized x position
    y: float    #Normalized y position
    succ: list                                      # liste des voisins infectieux
    status : int
    district : District    #n° du district BORDER = TRUE
    LOCKDOWN= True

    def __init__(self, x, y):
            self.x = x
            self.y = y
            self.district = compute_district(x, y)  # District ne peut être modifié car dépendant de x,y d'origine
            self.origin = (x,y)                     # idem pour la position d'origine car invariable
    
    def move(self):                                  # tout le monde bouge de la mÃªme faÃ§on
        dx,dy = np.random.normal(0,SIGMA, size=2)    #le mvt suit la loi normale
        x = self.x + dx
        y = self.y + dy
        if 0 <=x<= 1 and 0 <=y<= 1:                 # Conditions pour ne pas sortir de la map
            if BORDER and self.district == compute_district(x, y):  #  si district t = district t+1
                if LOCKDOWN and sqrt((x-self.origin[0])**2+(y-self.origin[1])**2)> MAX_HOME_DISTANCE: #interdiction de dÃ©passer le lockdown
                    return self
                self.x+=dx
                self.y+=dy
            if not BORDER:
                if LOCKDOWN and sqrt((x-self.origin[0])**2+(y-self.origin[1])**2)> MAX_HOME_DISTANCE : #il peut y avoir le lockdown et pas le border
                    return self
                self.x+=dx
                self.y+=dy

    def update(self):
        return self
    


#susceptible personne qui bouge aléatoirement       
class SusceptiblePerson(Person):
    state = SIRState.SUSCEPTIBLE       
    def update(self):
        infectedneighbor = False
        for people in self.succ:
            if people.state== SIRState.INFECTIOUS:
                infectedneighbor=True 
        if np.random.rand() < 0.5 and infectedneighbor==True:
            return InfectiousPerson (self.x, self.y)
        else:
            return self
    #si on a des voisins infecté on peut etre infecté avec une proba de 0,5
def voisin_infected_same_district(self): # pour chaque point dans le meme arrond que le point self d'interet regarde si il est infecte 
    for elt in self.succ: # pour chasue element dans la list des successeurs avant on regarde son etat  ( donc list doit prendr een compte stattut 
        if [elt].state == SIRState.INFECTIOUS: # le point a pour etat infecté 
            return Susceptibleperson(self.x,self.y)
        else :
            return self
def voisin_infected (self):   #voisin infectÃ©?
    for i in self.succ:
        if people[i].state == SIRState.INFECTIOUS:
            return True
    return False

        
class InfectiousPerson(Person):
    state = SIRState.INFECTIOUS
    age:int =0
    #si on a des voisins infecté on peut etre infecté avec une proba de 0,5
def voisin_infected_same_district(self): # pour chaque point dans le meme arrond que le point self d'interet regarde si il est infecte 
    for elt in self.succ: # pour chasue element dans la list des successeurs avant on regarde son etat  ( donc list doit prendr een compte stattut 
        if [elt].state == SIRState.INFECTIOUS: # le point a pour etat infecté 
                return InfectiousPerson(self.x,self.y)
        else :
                return self        
    def update(self):
        self.age+=1
        if np.random.rand() < GAMMA2 :
            return DeadPerson(self.x, self.y) #proba de mourir  après infection
        if self.age >= GAMMA1 : #si plus que 7 jours
            return RecoveredPerson(self.x, self.y)

        if BORDER:          #si le voisin infectÃ© est dans le mm district
            has_infected_neighbor = voisin_infected_same_district(self)

        else:               #sinon seulement si il a un voisins infectÃ©
            has_infected_neighbor = voisin_infected(self)

        if not has_infected_neighbor: #si pas de voisin contaminÃ©, pas de contamination
            return self

        if has_infected_neighbor:
            beta=BETA1  #proba de devenir infectieux
        
        if has_infected_neighbor and  np.random.random() < beta :
            return InfectiousPerson(self.x, self.y)
        else:
            return self
        
class RecoveredPerson(Person):
    state = SIRState.RECOVERED
    def update(self):
        if np.random.rand() < EPSILON :  #redeviend susceptible
            return SusceptiblePerson(self.x, self.y)
        else:
            return self #immunisé
        
class DeadPerson(Person):
    state = SIRState.DEAD
    def move(self):  #dead is dead
        pass

 
'''
Fonctions used to display and plot the curves
(you should not have to change them)
'''

def display_map(people, ax = None):
    x = [ p.x for p in people]
    y = [ p.y for p in people]
    h = [ p.state.name[0] for p in people]
    horder = ["S", "I", "R", "D"]
    ax = sns.scatterplot(x, y, hue=h, hue_order=horder, ax=ax)
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((0.0,1.0))
    ax.set_aspect(224/145)
    ax.set_axis_off()
    ax.set_frame_on(True)
    ax.legend(loc=1, bbox_to_anchor=(0, 1))


count_by_population = None
def plot_population(people, ax = None):
    global count_by_population

    states = np.array([p.state.value for p in people], dtype=int)
    counts = np.bincount(states, minlength=4)
    entry = {
        "Susceptible" : counts[SIRState.SUSCEPTIBLE.value],
        "Infectious" : counts[SIRState.INFECTIOUS.value],
        "Dead" : counts[SIRState.DEAD.value],
        "Recovered" : counts[SIRState.RECOVERED.value]
    }
    cols = ["Susceptible", "Infectious", "Recovered", "Dead"]
    if count_by_population is None:
        count_by_population = pd.DataFrame(entry, index=[0.])
    else:
        count_by_population = count_by_population.append(entry, ignore_index=True)
    if ax != None:
        count_by_population.index = np.arange(len(count_by_population)) / 24
        sns.lineplot(data=count_by_population, ax = ax)
        
'''
Main loop function, that is called at each turn
'''
def next_loop_event(t):
    print("Time =",t)

    # Move each person
    for p in people:
        p.move()

    update_graph(people)

    # Update the state of people
    for i in range(len(people)):
        people[i] = people[i].update()

    if t % FRAME_RATE == 0:
        fig.clf()
        ax1, ax2 = fig.subplots(1,2)
        display_map(people, ax1)
        plot_population(people, ax2)
    else:
        plot_population(people, None)

'''
donne les personnes infectieuses voisines
'''
#faire une liste propre a cahque personne qui ont des voisins et voir le nombre de infectious people dans ces voisins 
UNSEEN = 0; DONE = 1
def update_graph(people):  #update liste des successeurs infectieux
    for i in (people):
        i.succ=[]
        for j in (people):
            D=((i.x-j.x)**2+(i.y-j.y)**2)**0.5 #cf théorème de Pythagore
            if i is not j and D <= SOCIAL_DISTANCE : #si le respect des distances n'est pas respecté 
                i.succ.append(j)  #liste de voisins infectieux
        

'''
Function that crate the initial population
'''
def create_data():
    # This creates a susceptible person located at (0.25,0.5)
    # and an infectious person located at (0.75,0.5)
   a=[]
   for i in range(DENSITY):
        if np.random.random()< I0:          #créer une personne INFECTIOUS avec une probabilité
            a.append(InfectiousPerson(np.random.rand(),np.random.rand()))
        else :
            a.append(SusceptiblePerson(np.random.rand(),np.random.rand()))
   return a                                #liste de toutes les personnes de l'étude

def create_data_test():
    S = [ SusceptiblePerson(0.5, 0.25) for i in range(100) ]
    I = [ InfectiousPerson(0.5, 0.76) for i in range(100) ]
    return S + I

       


import matplotlib.animation as animation
people = create_data()

fig = plt.figure(1)
duration = 20 # in days
anim = animation.FuncAnimation(fig, next_loop_event, frames=np.arange(duration*24), interval=100, repeat=False)

# To save the animation as a video
#anim.save("simulation.mp4", fps=5, dpi=100, writer="ffmpeg")

plt.show()
