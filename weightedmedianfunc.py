""""
Annabelle Recinos
created: 1/17/21
finalized: 5/3/21

weightedmedianfunc.py

This program caluclates the weighted median of scores, passed in from a list.
Once the weighted median haas been calculated then a graph is generated based
on the data. This is to help users understand the data and calculations.

will need to import the following libraries
import numpy as np
import matplotlib.pyplot as plt 

"""
def wm(nscores,dist):  
    #get weights of each score
    weights = [1/ x for x in dist]
    
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
    
    #get upper bound and element place of upper bound
    if lb != middle:
    	ub = lb + weights[low] + weights[low+1]
    	high = low + 1
    	uscore = nscores[high]
    	lscore = nscores[low]
    	d = ub - lb
    	t = ub - middle
    	wm = (uscore*((d-t)/d) + lscore*(t/d))
    else:
    	ub = lb
    	high = low
    	wm = lscore
    
    print("Weights")
    print("   Upper Bound:",round(ub,5))
    print("   Lower Bound:",round(lb,5))
    print("   Middle:", round(middle,5))
    print("Scores")
    print("   Upper Bound:",uscore)
    print("   Lower Bound:",lscore)
    print("   Weighted Median:",round(wm,2))
    
    
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