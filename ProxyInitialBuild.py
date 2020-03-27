# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:04:26 2020

@author: ymamo
"""

import ProxyAgent as PA
import ProxyCompany as PC
import copy


def agents_to_company(model, i, j, num_agents_left, co):
    A = PA.ProxyAgent(num_agents_left, model, co)
    num_agents_left -= 1
    model.ml.add(A)
    model.ml.add_link([(co,A)])
    ''' Add all agents row wise from top left to bottom right '''
    if model.grid.width > 1 and model.grid.height > 1:
        x = (i*10+j) % model.grid.width
        y = int((i*10+j)/model.grid.height)
        model.grid.place_agent(A, (x, y))
    
        
    return num_agents_left



def Build_Multi_Layer_World(model): 

    '''
    Purpose: Helper function to build more companies and employees
    
    Parameters: The model
    
    Builds company and associated agents
    
    Connections will be created after companies are built. 
    
    '''
    num_companies = int(model.num_agents/10)
    num_agents_per = int(model.num_agents/10)
    num_agents_left = copy.copy(model.num_agents)
    company_count = 1
    
        
    ''' Create agents, assign to companies and place on the grid '''
    for i in range(num_companies): 
        name = "Company_" +str(i)
        C = PC.ProxyCompany(name, model)
        model.ml.add(C)
        if num_agents_left > num_agents_per:
            for j in range(num_agents_per):
                num_agents_left = agents_to_company(model, i, j, num_agents_left,C)
                
        else:
            for j in range(num_agents_left):
                num_agents_left = agents_to_company(model, i, j, num_agents_left, C)
                
## Build cmpanies and givconstraints based on company flexibility
                