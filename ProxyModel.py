# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:10:01 2020

@author: ymamo
"""

import numpy as np
import random
#Helper module to help build agents and assign to companies
import ProxyInitialBuild as PIB


from mesa import Model
#from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import multilevel_mesa as mlm

from ProxyAgent import ProxyAgent
from ProxyCompany import ProxyCompany
import Collector_Functions as cf


class ProxyModel(Model):
    """
    Model class
    - initialize model (agents on grid, time)
    - step model (implement selection/evolution, collect data)
    """
    def __init__(self,
                 data_collect_interval,
                 width, height,
                 practice_mutation_rate,
                 talent_sd, competition,
                 numAgents, selection_pressure,
                 survival_uncertainty,
                 goal_scale,
                 goal_angle):
        self.data_collect_interval = data_collect_interval
        self.num_agents = numAgents
        self.selection_pressure = selection_pressure
        self.grid = SingleGrid(width, height, True)  # toroidal (all ends rap)
        self.talent_sd = talent_sd
        self.practice_mutation_rate = practice_mutation_rate
        self.competition = competition
        self.survival_uncertainty = survival_uncertainty
        self.goal_scale = goal_scale
        #self.schedule = RandomActivation(self)
        self.ml = mlm.MultiLevel_Mesa(self)
        #!!! A cheat to get the agent list update ml mesa to prevent data collector error 
        self.schedule = self.ml
        self.running = True
        self.time = 0
        #!!! Cheat becuase ml mesa is not compatible with batch runner
        self.schedule.steps = self.time
        self.goal_angle = goal_angle
 
        
        PIB.Build_Multi_Layer_World(self)
        
                
 
        self.datacollector = DataCollector(
                model_reporters={"mean_proxy_value": cf.compute_mean_proxy_value,
                                 "mean_goal_value": cf.compute_mean_goal_value,
                                 "mean_goal_oc": cf.compute_mean_goal_oc,
                                 "mean_effort": cf.compute_mean_effort,
                                 "mean_utility": cf.compute_mean_utility,
                                 "mean_practice": cf.compute_mean_practice},
                agent_reporters={"Proxy": cf.get_agent_proxy ,
                                 "Goal": cf.get_agent_goal,
                                 "Goal_oc": cf.get_agent_goal_oc,
                                 "Utility": cf.get_agent_utility,
                                 "Effort": cf.get_agent_effort,
                                 "Practice": cf.get_agent_practice,
                                 "Genealogy": cf.get_agent_child_of,
                                 "Talent": cf.get_agent_talent,
                                 "Type": cf.get_agent_type})
 
    def kill_and_replace(self):
        """ recompute rank with chosen effort levels
        randomly kill losers with probability = sp and
        replace with offspring from random winner
        (new agents 1. inherit their practice and effort from the parent,
        2. draw new random talent,
        3. take the location & ID from the dead agent to facilitate display.
        Deaths, births and genealogy are stored in "Genealogy")"""
        #ml-change switch agents reference
        #agents = self.schedule.agents
        agents = list(self.ml.agents_by_type[ProxyAgent].values())
        proxies = list(n.proxy for n in agents)
        rel_surv_thresh = self.competition
        ordered = np.sort(proxies)
        survival_threshold = ordered[int(rel_surv_thresh*len(proxies))-1]
        # print(survival_threshold)
        
        potential_losers = list(losers for losers in agents if losers.proxy <= survival_threshold)
        potential_winners = list(losers for losers in agents if losers.proxy >= survival_threshold)
            
        for potential_loser in potential_losers:
            if np.random.rand() < self.selection_pressure:
                loser = potential_loser
                winner = random.choice(potential_winners)
 
                ''' offspring: here the loser becomes the offspring of the winner '''
                loser.effort = winner.effort
                loser.practice = np.random.normal(winner.practice,
                                                  self.practice_mutation_rate)
                loser.talent = np.random.normal(10, self.talent_sd)
 
                ''' practice within 360° '''
                if loser.practice > np.pi*2:
                    loser.practice = loser.practice - np.pi*2
     
                ''' no negative talent '''
                if loser.talent < 0:
                    loser.talent = 0.01
                loser.child_of = winner.unique_id
 
    def step(self):
        ''' adjust effort levels in random order '''
        #ml-change switch step reference
        #self.schedule.step()
        self.ml.step()
        self.time += 1
        #!!! Cheat becuase ml meas is not compatible with batchrunner
        self.schedule.steps = self.time
        self.kill_and_replace()
        d = self.data_collect_interval
        #print (d, self.time, self.schedule.steps)
        if self.time % d == 0:
            self.datacollector.collect(self)