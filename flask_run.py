#!/usr/bin/env python

from tkinter import W
import numpy as np
import pickle
# WARNING I AM FILTERING WARNINGS BECUASE PATHOS DOESN'T LIKE THEM
import warnings
warnings.filterwarnings("ignore")

from EvolSearch import EvolSearch
from Flask_Evo import flask_fitness
from functools import partial


use_best_individual = False
if use_best_individual:
    with open("best_individual", "rb") as f:
        best_individual = pickle.load(f)

########################
# Parameters
########################
# Init Params
name = "test"
N = 3
types = 20
init_eq_steps = 500
timesteps = 1000

# Seasonal Params
seasonalityN = True
s_length = 8000
s_intensity = 0.5
offset = s_length/N
barren = 0.0

# Chemostat & Life History Params
inflow = 5
outflow = 0.2
maxcons = 0.25
col = 0.001
mutation = 0.08

########################
# Evolve Solutions
########################

pop_size = 200
genotype_size = 3*N*types 

evol_params = {
    "num_processes": 100,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": partial(flask_fitness, types=types, N=N, 
                                s_length=s_length, s_intensity=s_intensity, offset=offset,
                                inflow=inflow, outflow=outflow,barren=barren,maxcons=maxcons,col=col,mutation=mutation,
                                seasonalityN=seasonalityN,timesteps=timesteps),  
                                # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generation
    "mutation_variance": 0.1,  # mutation noise added to offspring.
}

initial_pop = []
for i in range(pop_size):
    type_list = []
    for t in range(types):
        genes = 2*N
        genome = list(2*(np.random.sample(genes))-1)
        for i in range(N):
            genome.append(np.random.sample())
        type_list.append(genome)
    type_list = [item for sublist in type_list for item in sublist]
    initial_pop.append(type_list)

#if use_best_individual:
#    initial_pop[0] = best_individual["params"]

evolution = EvolSearch(evol_params, initial_pop)

save_best_individual = {
    "params": None,
    "best_fitness": [],
    "mean_fitness": [],
}

for i in range(20):
    evolution.step_generation()
    
    save_best_individual["params"]= evolution.get_best_individual()
    
    save_best_individual["best_fitness"].append(evolution.get_best_individual_fitness())
    save_best_individual["mean_fitness"].append(evolution.get_mean_fitness())

    print(
        len(save_best_individual["best_fitness"]), 
        save_best_individual["best_fitness"][-1], 
        save_best_individual["mean_fitness"][-1]
    )

    with open("best_individual", "wb") as f:
        pickle.dump(save_best_individual, f)