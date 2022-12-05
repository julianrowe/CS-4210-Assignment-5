#-------------------------------------------------------------------------
# AUTHOR: Julian Rowe
# FILENAME: association_rule_mining.py
# SPECIFICATION: Read reatil_dataset.csv to find strong rules related to 
# supermarket products satisfying minsup = 0.2 and minconf = 0.6
# FOR: CS 4210- Assignment #5
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
for index, row in df.iterrows():

    labels = {'Meat': 0, 'Diaper': 0, 'Bread': 0, 'Eggs': 0, 'Wine': 0, 'Bagel': 0, 'Pencil': 0, 'Milk': 0, 'Cheese': 0}
    for item in itemset:
        labels[item] = 1
    for item in row:
        if item in labels:
            labels[item] = 0

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below

for rule in rules.iterrows():
    antecedent = ', '.join(rule[1][0])
    consequent = ', '.join(rule[1][1])
    print(antecedent + ' -> ' + consequent)

    support = rule[1][4]
    print('Support:', support)

    confidence = rule[1][5]
    print('Confidence:', confidence)

    conseFrozen = rules.loc[i]['consequents']
    temp = list(conseFrozen)
    supportCount = sum(row[temp[0]] for row in encoded_vals)
    prior = supportCount / len(encoded_vals)
    print('Prior:', prior)

    gain = str(100 * (confidence - prior) / prior)
    print('Gain in Confidence:', gain)
    print("\n")

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()