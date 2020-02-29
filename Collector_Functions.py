# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:08:24 2020

@author: ymamo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:48:08 2017
 
@author: Oliver Braganza
 
proxyeconomics model
The model is based on mesa framework. Agent properties and computations are in
the Agent class, System level properties and computations in the Model class.
"""
 
#from ProxyModel import ProxyModel
import numpy as np
 
''' Functions computing model level readouts for data collection '''
 
 
def compute_mean_proxy_value(model):
    """ Returns the mean total proxy value across agents """
    #ml-change switch schedule reference
    #proxy_values = [agent.proxy for agent in model.schedule.agents]
    proxy_values = [agent.proxy for agent in model.ml.agents if agent.type == "individual"]
    return np.mean(proxy_values)
 
 
def compute_mean_goal_value(model):
    """ Returns the mean goal value across agents """
    #ml-change switch schedule reference
    #goal_values = [agent.goal for agent in model.schedule.agents]
    goal_values = [agent.goal for agent in model.ml.agents if agent.type == "individual"]
    return np.mean(goal_values)
 
 
def compute_mean_goal_oc(model):
    """ Returns the mean independent goal component across agents """
    #ml-change switch schedule reference
    #goal_oc = [agent.goal_oc for agent in model.schedule.agents]
    goal_oc = [agent.goal_oc for agent in model.ml.agents if agent.type == "individual"]
    return np.mean(goal_oc)
 
 
def compute_mean_effort(model):
    """ returns the mean effort across agents """
    #ml-change switch schedule reference
    #effort_values = [agent.effort for agent in model.schedule.agents]
    effort_values = [agent.effort for agent in model.ml.agents if agent.type == "individual"]
    return np.mean(effort_values)
 
 
def compute_mean_utility(model):
    """ returns the mean utility across agents """
    #ml-change switch schedule reference
    #utility_values = [agent.utility for agent in model.schedule.agents]
    utility_values = [agent.utility for agent in model.ml.agents if agent.type == "individual"]
    return np.mean(utility_values)
 
 
def compute_mean_practice(model):
    """ returns the mean practice angle across agents """
    #ml-change switch schedule reference
    #pr_vals = [agent.practice for agent in model.schedule.agents]
    pr_vals = [agent.practice for agent in model.ml.agents if agent.type == "individual"]
    return np.arctan2(np.mean(np.sin(pr_vals)), np.mean(np.cos(pr_vals)))

####################################################
#                                                  #
#              Agent  Collectors                   #
#                                                  #
####################################################      

def get_agent_proxy(agent):
    if agent.type == "individual":
        return agent.proxy
    else:
        return "none"
    
def get_agent_goal(agent):
    if agent.type == "individual":
        return agent.goal
    else:
        return agent.company_policy
    
def get_agent_goal_oc(agent):
    if agent.type == "individual":
        return agent.goal_oc
    else:
        return "none"

def get_agent_utility(agent):
    if agent.type == "individual":
        return agent.utility
    else:
        return "none"

def get_agent_effort(agent):
    if agent.type == "individual":
        return agent.effort
    else:
        return "none"

def get_agent_practice(agent):
    if agent.type == "individual":
        return agent.practice
    else:
        return "none"
  

def get_agent_child_of(agent):
    if agent.type == "individual":
        return agent.child_of
    else:
        return "none"                               
    
def get_agent_talent(agent):
    if agent.type == "individual":
        return agent.talent
    else:
        return "none"  
    
def get_agent_type(agent):
    return agent.type


    
                            
 
 
