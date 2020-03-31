# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:05:48 2020

@author: ymamo
"""
import numpy as np
from mesa import Agent



class ProxyAgent(Agent):
    """
    Agent class
    - initialize agents (practice, effort,..
    - step agents (optimize effort/practice to maximize utility)
    """
    def __init__(self, unique_id, model, co):
        super().__init__(unique_id, model)
        self.practice = np.random.uniform(co.company_policy-(model.goal_angle/co.company_flexibility),
                                          co.company_policy+(model.goal_angle/co.company_flexibility))
        #tpike: what is going on here this means that self.goal is always 0???
        #self.practice = self.model.goal_angle
        self.talent = np.random.normal(10, self.model.talent_sd)
        if self.talent < 0:
            self.talent = 0.01
        self.effort = 0
        self.proxy = np.cos(self.practice) * self.effort
        self.goal = np.cos(self.model.goal_angle - self.practice) * self.effort
        #Oliver: not sure what your asking above. self.goal is calculated here from practice and effort
        self.goal_oc = np.sin(self.practice) * self.effort
        self.goal_scale = self.model.goal_scale
        self.utility = np.nan
        self.child_of = self.unique_id
        #ml-change added for multi_levels
        self.type = "individual"
        self.connection = co.unique_id
 
    def step(self):
        """ Actions to perform on each time step """
        self.optimize_effort()
 
    def optimize_effort(self):
        """ Heuristic to optimize effort level (and potentially practice):
        Vary effort by test_list and check if utility increases.
        Utility has 3 components:
        1. proxy value (extrinsic)
        utility/disutility derived from prospect of surviving competition
        computed from own relative proxy-rank within population
        2. goal value (intrinsic)
        utility/disutility derived from contributing to the societal goal
        3. effort cost
        disutility due to effort expenditure
        effort cost = effort^2 /talent
 
        If agents have agency over the practice angle, they similarly optimize
        by going through test_list at every angle in angle_list.
        """
        test_list = [-10, -5, -1, -0.5, -0.1,
                     0, 0.1, 0.5, 1, 5, 10]
 
        ''' agency >0 introduces agency over the practice angle '''
        agency = 0  # 0 means no agency; 1 means full agency
        angle_list = [self.practice]
        if np.random.rand() < agency:
            ''' social learning '''
#            angle_list = neighbor_practices
            ''' individual learning (gaming) '''
            own_practice = self.practice/np.pi*180
            change_angle = [-5, -1, 0, 1, 5]
            angle_list = [own_practice-x for x in change_angle]
            angle_list = np.deg2rad(angle_list)
 
 
        def get_prospect(self):
            """ calculates the utility/disutility from the prospect of winning/
            loosing competition """
 
            #''' list of neighbors proxy performances '''
            # neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=2)
            # proxies = list(n.proxy for n in neighbors)
            #ml-change switch to ml scheduler
            agents = self.model.schedule.agents
            #agents = self.model.ml.agents
            proxies = list(n.proxy for n in agents if n.type == "individual")
            self.proxy = np.cos(self.practice) * self.effort
            own_proxy = self.proxy
 
            rel_surv_thresh = self.model.competition
            ordered = np.sort(proxies)
            survival_threshold = ordered[int(rel_surv_thresh*len(proxies))-1]
            if self.model.competition > 0:
                ''' McDermott Prospect '''
                # prospect = ps * ss.erf((own_proxy-survival_threshold)/su)
                ''' Kahneman Tversky Prospect '''
                prospect = abs(own_proxy-survival_threshold)**0.88
                ''' Step Prospect '''
                # prospect = 1
            else:
                prospect = 0  # no competition
            if (own_proxy-survival_threshold) < 0:  # Kahneman Tversky loss aversion
                prospect = -abs(prospect) * 2.25
                # prospect = -1
            return prospect
 
        def get_utility(self, prospect):
            ''' utility function '''
            self.goal = np.cos(self.model.goal_angle - self.practice) * self.effort
            gsc = self.goal_scale
            e = self.effort
            t = self.talent
 
            utility = prospect + gsc*self.goal - (e**2)/t
 
            return utility
 
        old_effort = self.effort
        new_effort = 0
        new_practice = self.practice
        max_utility = -1000
        for test_angle in angle_list:
            for test_effort in test_list:
                self.effort = old_effort + test_effort
                self.practice = test_angle
                if self.effort > 0:
                    prospect = get_prospect(self)
                    utility = get_utility(self, prospect)
                    if np.isnan(utility):
                        print('error: utility is nan')
                    if utility > max_utility:
                        max_utility = utility
                        new_effort = self.effort
                        new_practice = self.practice
 
        self.utility = max_utility
        self.effort = new_effort
        self.practice = new_practice
        self.oldproxy = self.proxy
        self.proxy = np.cos(self.practice) * self.effort 
        self.goal = np.cos(self.model.goal_angle - self.practice) * self.effort
        self.goal_oc = np.sin(self.practice) * self.effort