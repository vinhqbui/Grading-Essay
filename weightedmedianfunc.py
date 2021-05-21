"""
Annabelle Recinos
created: 1/17/21
finalized: 5/4/21

weightedmedianfunc.py

This code is a function that calculates the weighted median. A list of scores and 
their corresponding weights are passed in as arguments. Once the weighted median
is calculated than a graph is generated. This is help users visualize the data
and the calculations.  
"""

import numpy as np
import matplotlib.pyplot as plt

def weighted_median(nscores,dist, printMode=False):
    wm = 0
    #get weights of each score
    for i in range(len(dist)):
        if dist[i] == 0:
            dist[i] = 0.000000000000000000001
    weights = [1.0/ x for x in dist]
    
    #sum weights to find middle
    middle = 0
    for x in range(0, len(weights)):
        middle += weights[x]
    
    #get lower bound and element place of lower bound
    lb = weights[0] 
    low = 0 
    while lb < middle:
        lb += weights[low] + weights[low+1]
        low += 1
    
    if lb > middle:
        lb -= (weights[low]+weights[low-1])
        low -= 1
        
    ub = 0.0
    #get upper bound and element place of upper bound
    if lb != middle:
    	ub = lb + weights[low] + weights[low+1]
    	high = low + 1
    	uscore = nscores[high]
    	lscore = nscores[low]
    	d = ub - lb
    	t = ub - middle
    	if d == 0:
            wm = ub
    	else:
            wm = (uscore*((d-t)/d) + lscore*(t/d))
    else:
    	ub = lb
    	high = low
    	wm = nscores[low]
    
    #this while will allow for users to either print the results if they want
    if printMode:
        try:
            print("Do you want the results printed?")
            print("Enter: yes or no")
            resp1 = input()
            print("\n\n")
    
            if resp1 == "yes":
                print("Results:")
                print("Weights")
                print("   Upper Bound:",round(ub,5))
                print("   Lower Bound:",round(lb,5))
                print("   Middle:", round(middle,5))
                print("Scores")
                print("   Upper Bound:",uscore,"%")
                print("   Lower Bound:",lscore,"%")
                print("   Weighted Median:",round(wm,2))

            else: 
                print("Please input a valid response: yes or no.")
        except Exception as e:
                print(e)
    
    #this while will allow users to print the graph if they want
    if printMode:
        try:
            print("Would you like the graph of the weighted median printed?")
            print("Enter: yes or no")
            resp2 = input()
            print("\n\n")
    
            if resp2 == "yes":
                #position gets the starting egdes of each bar in the graph
                position = np.cumsum([2*x for x in [0] + weights[:-1]]) 
                diameter = [2*x for x in weights] #gets width of each bar
                
                #this codes and styles the graph to be printed to users
                plt.bar(position,nscores,diameter,color = "#1E90FF", edgecolor = "black", align = "edge")
                plt.axhline(wm, color = "#006400", label = "weighted median")
                plt.axvline(middle, color ="red", label = "middle")
                plt.axvline(ub, color ="#2F4F4F", linestyle ="dashdot", label= "upper bound" )
                plt.axvline(lb, color ="#2F4F4F", linestyle ="dashed", label= "lower bound")
                plt.margins(x=0)
                plt.xlabel("Weights")
                plt.ylabel("Scores")
                plt.title("Weighted Median")
                plt.legend(bbox_to_anchor=(1.05,.5), ncol= 1)
                plt.plot([lb,ub],[lscore, uscore], color ="black")
                plt.show()
            else:
                 print("Please input a valid response: yes or no.")
        except Exception as e:
                print(e)     
    
    
    return wm