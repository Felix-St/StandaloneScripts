# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:44:57 2021

@author: Felix
"""

"""
    Short program that one can use to visualize the Odds of getting a specific item from
    a cs go case / collection. Simply call calculateRarity(X), X being the amount of rarities.
    For example a normal cs go case has 5 rarities. Thus call calculateRarity(5).
    
"""



def calculateRarity(rarities_amount):
    rarities = range(1,rarities_amount+1)
    percentages = []
    temp_prob = 1
    factor = 0
    for i,rarity in enumerate(rarities):
        for i in range(rarity-i,0,-1):
            # This part actually used dynamic programming to keep track
            # of the factor, so that a recaculation is not necessary.
            factor += 5**(rarity-i)
        temp_prob = 1/factor 
        percentages.append(temp_prob)
    return percentages 
        
        
        