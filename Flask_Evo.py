#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Flask Model

@author: eden
"""
##### IMPORTS #####
import numpy as np
import math

def flask_fitness(type_list, types, N, s_length, s_intensity, offset, barren, inflow, outflow, maxcons, col, mutation, seasonalityN, timesteps):

    class flask:
        def __init__(self, type_list, types, N, s_length, s_intensity, offset, barren, inflow, outflow, maxcons, col, mutation, seasonalityN, timesteps):
            
            ##### CONSTANTS #####
            # Env. Composition
            self.N = N
            self.timesteps = timesteps
            self.seasonality = seasonalityN
            self.nutrient_vec = np.zeros((timesteps,self.N))
            self.cycling_vec = np.zeros((timesteps,self.N))
            self.inflow_vecN = np.zeros((timesteps,self.N))
            self.pop_vec = np.zeros((timesteps,self.N))
            self.trait_vec = [[0]*timesteps]
            self.trait_list = list()
            
            ##### INITS #####
            self.nutrients = np.zeros(self.N)
            self.nut_demand = np.zeros(self.N)
            self.feed_vec = np.zeros(self.N)
            self.population = list()
            self.types = types

            self.type_list = np.reshape(type_list, (types, 3*N))
            
            self.init_eq_steps = 500
            self.steps = 0
            
            ## Life History
            self.r_thresh = 120
            self.p_mut = mutation ## 0 or 0.01
            self.d_thresh = 50
            self.d_chance = 0.002
            self.living_cost = col
            self.bio_high = 100
            self.bio_low = 70
            
            ## Consumption & Flux
            self.cons_max = maxcons
            self.nutrient_inflow = inflow ## 0 to 150
            self.nutrient_outflow = outflow ## 0.01 to 0.25
            ## Lower -> Longer
            self.s_length = s_length
            self.s_intensity = s_intensity
            self.offset = offset
            self.barren = barren
            self.ssnN = self.s_intensity*self.nutrient_inflow*(1 - self.barren)
            
        ##### FUNCTIONS #####
        
        ##### STEP #####
        def step(self):
            if self.steps == self.init_eq_steps:
                self.start_Population()
            self.influx()
            self.micro_Feed()
            self.micro_Excrete()
            self.pop_Update()
            self.outflux()
            self.trait_Update()
            self.nutrient_vec[self.steps] = list(self.nutrients)
            self.cycling_vec[self.steps] = list(self.feed_vec/self.nutrient_inflow)
            self.pop_vec[self.steps] = len(self.population)
            self.steps = self.steps + 1

        def trait_Update(self):
            for p in self.population:
                if any(l == p.b_loci[:] for l in self.trait_list):
                    self.trait_vec[self.trait_list.index(p.b_loci[:])][self.steps] += 1
                else:
                    self.trait_list.append(p.b_loci[:])
                    newrow = [0]*self.timesteps
                    newrow[self.steps] = 1
                    self.trait_vec.append(newrow)
        
        def start_Population(self):
            for t in self.type_list:
                for i in range(5):
                    biomass = np.random.randint(self.bio_low,self.bio_high)
                    m = microbe(self,biomass,t)
                    self.population.append(m)
        
        def seasonN(self):
            inflow_vec = np.zeros(self.N)
            for i in range(len(inflow_vec)):
                inflow_vec[i] = (self.s_intensity*self.nutrient_inflow)*math.sin(((self.steps+self.offset*i)/self.s_length) * 2 * math.pi) + self.ssnN
                if inflow_vec[i] < 0:
                    inflow_vec[i] = 0
            return(inflow_vec)

        def influx(self):
            if self.seasonality == True:
                inflowN = self.seasonN()
            else:
                inflowN = [self.nutrient_inflow]*self.N
            self.inflow_vecN[self.steps] = list(inflowN)
            for n in range(self.N):
                self.nutrients[n] = self.nutrients[n] + inflowN[n]
            
        def outflux(self):
            for n in range(self.N):
                self.nutrients[n] = self.nutrients[n] - (self.nutrient_outflow*self.nutrients[n])
                if self.nutrients[n] < 0:
                    self.nutrients[n] = 0
            
        def pop_Update(self):
            for p in self.population:
                d_check = False
                ## Check for Death Threshold
                if p.biomass <= self.d_thresh:
                    self.population.remove(p)
                    d_check = True
                ## Random Chance of Death/Washout
                if d_check == False:
                    if np.random.sample() < self.d_chance:
                        self.population.remove(p)
                ## Check for Repro Threshold
                if p.biomass >= self.r_thresh:
                    p.biomass = p.biomass/2
                    child = microbe(self,p.biomass,p.genome)
                    for i in range(self.N):
                        ## MUTATION
                        if np.random.sample() < self.p_mut:
                            child.b_loci[i] = np.random.sample()
                    self.population.append(child)
                
        def micro_Demand(self):
            demands = np.zeros(self.N)
            for i in range(len(demands)):
                for p in self.population:
                    demands[i] = demands[i] + p.consumption[i] * (1 - p.b_loci[i]**4/(p.b_loci[i]**4+self.nutrients[i]**4))
            return demands
        
        def micro_Feed(self): 
            demands = self.micro_Demand()
            feed_vec = np.zeros(self.N)
            for p in self.population:
                total_feed = 0
                total_excrete = 0
                for i in range(len(demands)): 
                    ## Consumption preference is same as consumption efficiency
                    if demands[i] > self.nutrients[i]:
                        p_dem = self.cons_max * p.consumption[i] * (1 - p.b_loci[i]**4/(p.b_loci[i]**4+self.nutrients[i]**4))
                        ## Scale to availability
                        feed = (p_dem/demands[i])*self.nutrients[i]*p.consumption[i]
                        excrete = (p_dem/demands[i])*self.nutrients[i]*(1-p.consumption[i])
                        total_feed = total_feed + feed 
                        total_excrete = total_excrete + excrete
                        feed_vec[i] = feed_vec[i] + feed
                    else:
                        ## No need to scale
                        feed = self.cons_max * p.consumption[i] * (1 - p.b_loci[i]**4/(p.b_loci[i]**4+self.nutrients[i]**4))
                        excrete = self.cons_max * (1-p.consumption[i]) * (1 - p.b_loci[i]**4/(p.b_loci[i]**4+self.nutrients[i]**4))
                        total_feed = total_feed + feed 
                        total_excrete = total_excrete + feed 
                        feed_vec[i] = feed_vec[i] + feed 
                p.excrete = total_excrete
                ## GROWTH RATE FUNCTION
                p.biomass_change = total_feed - self.living_cost
                p.biomass = p.biomass + p.biomass_change
            self.nutrients[0:self.N] = self.nutrients[0:self.N] - feed_vec
            self.feed_vec = feed_vec
                
        def micro_Excrete(self):
            for p in self.population:
                ex_vec = []
                for e in p.excretion:
                    ex_vec.append(e * p.excrete)
                self.nutrients[0:self.N] = self.nutrients[0:self.N] + ex_vec  
                    
    class microbe:
        def __init__(self, flask, biomass, genome):
            # Inits
            self.biomass = biomass
            self.biomass_change = 0
            self.genome = genome.copy()
            ## Consumption & Excretion
            self.c_loci = genome[0:flask.N]
            self.e_loci = genome[flask.N:2*flask.N]
            net_cons = np.abs(self.c_loci) - np.abs(self.e_loci)
            consumption = list()
            excretion = list()
            for i in net_cons:
                if i < 0:
                    consumption.append(0)
                    excretion.append(i)
                else:
                    consumption.append(i)
                    excretion.append(0)
            sum_c = sum(consumption)
            sum_e = sum(excretion)
            if sum_c != 0:
                for i in range(len(consumption)):
                    consumption[i] = abs(consumption[i]/sum_c)
            if sum_e != 0:
                for i in range(len(excretion)):
                    excretion[i] = abs(excretion[i]/sum_e)
            for i in range(len(consumption)):
                if consumption[i] != 0 and excretion[i] != 0:
                    consumption = np.zeros(flask.N)
            
            ## Behavior
            self.b_loci = genome[2*flask.N:3*flask.N]
            for i in range(len(consumption)):
                if consumption[i] == 0:
                    self.b_loci[i] = 0
            sum_b = sum(self.b_loci)
            if sum_b != 0:
                for i in range(len(self.b_loci)):
                    self.b_loci[i] = self.b_loci[i]/sum_b
            self.b_loci = self.b_loci.tolist()
                    
            self.consumption = consumption
            # print("My consumption is {}").format(self.consumption)
            self.excretion = excretion
            # print("My excretion is {}").format(self.excretion)
            self.behavior = self.b_loci
            # print("My behavior is {}").format(self.behavior)
            
            self.excrete = 0
            self.cons_max = 0

    f = flask(type_list,types,N,s_length,s_intensity,offset,barren,inflow,outflow,maxcons,col,mutation,seasonalityN,timesteps)
    for t in range(timesteps):
        f.step()
    ## Fitness evaluation is average community population over the last N
    return np.mean(f.pop_vec[-20:])

