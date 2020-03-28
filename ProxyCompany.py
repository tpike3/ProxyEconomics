# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:26:55 2020

@author: ymamo
"""

import numpy as np
from mesa import Agent



class ProxyCompany(Agent):
    """
    Agent class
    - initialize agents (practice, effort,..
    - step agents (optimize effort/practice to maximize utility)
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.company_policy = np.random.uniform(0, self.model.goal_angle)
        self.company_flexibility = np.pi/10
        #!!! what is going on here this means that self.goal is always 0???
        #oliver: initializing self.practice at the goal-angle means that agents start off completely oriented toward the goal, i.e. with large self.goal
        #self.practice = self.model.goal_angle
        self.talent = np.random.normal(10, self.model.talent_sd)
        if self.talent < 0:
            self.talent = 0.01
        self.effort = 0
        self.company_proxy = np.cos(self.company_policy) * self.effort
        self.goal = np.cos(self.model.goal_angle - self.company_policy) * self.effort
        self.goal_oc = np.sin(self.company_policy) * self.effort
        self.goal_scale = self.model.goal_scale
        self.utility = np.nan
        self.child_of = self.unique_id
        #ml-change added for multi_levels
        self.type = "company"