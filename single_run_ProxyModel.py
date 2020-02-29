# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:52:43 2020

@author: ymamo
 x"""

import ProxyModel as pm
import numpy as np
import pandas as pd
import scipy.stats as stats



finalStep = 10
variable_parameters = {"competition": np.linspace(0.1, 0.9, 9)}
parameters = {"data_collect_interval": 1,
              "width": 10, "height": 10,
#              "competition": np.linspace(0.1, 0.9, 9),
              "numAgents": 100,
              "talent_sd": 1,
              "goal_scale": 2,
              "goal_angle": np.pi/4,
              "selection_pressure": 0.1,
              "practice_mutation_rate": np.pi/90,
              "survival_uncertainty": 3}    # currently not in model (KT)
print(parameters)


model = pm.ProxyModel(parameters["data_collect_interval"],
                      parameters["width"],
                      parameters["height"],
                      parameters["practice_mutation_rate"],
                      parameters["talent_sd"],
                      variable_parameters["competition"][0],
                      parameters["numAgents"],
                      parameters["selection_pressure"],
                      parameters["survival_uncertainty"],
                      parameters["goal_scale"],
                      parameters["goal_angle"]
                      )
for i in range(finalStep):
    model.step()
    print (i)