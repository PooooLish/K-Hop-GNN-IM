import time
import pandas as pd
import numpy as np

def IC(G,S,mc=100):
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:
            temp = G.loc[G['source'].isin(new_active)]
            # For each newly active node, find its neighbors that become activated
            targets = temp['target'].tolist()
            ic = temp['weight'].tolist()
            # Determine the neighbors that become infected
            coins  = np.random.uniform(0,1,len(targets))
            choice = [ic[c]>coins[c] for c in range(len(coins))]
            #sum(choice)

            new_ones = np.extract(choice, targets)

            # Create a list of nodes that weren't previously activated
            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active
           
        spread.append(len(A))
        
    return(np.mean(spread))
