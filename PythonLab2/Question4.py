'''
Using Numpy create random vector of size 15 having only Integers in the range 0 -20.
Write a program to find the most frequent item/value in the vector list.

Sample input:
[1,2,16,14,6,5,9,9,20,19,18]

Sample output:
Most frequent item in the list is 9
'''
import numpy as np

#set up variable
highestFreq = {"number": set(), "Freq": 0} #use set so no duplicates

randomVector = list(np.random.randint(1,20, size=15)) #Create random vector of size 15 between 1 and 20

for i in randomVector: #iterate through randomVector
    if randomVector.count(i)>highestFreq["Freq"]: #if count is higher than highest freq
        highestFreq["number"] = set() #initialize set or empty old set
        highestFreq["Freq"] = randomVector.count(i)#put in freqency
        highestFreq["number"].add(i) #put in new number
    elif randomVector.count(i) == highestFreq["Freq"]:
        highestFreq["number"].add(i) # if number has same frequency as previous then add it

#code to make a sentence when printing multiple variables of same freqency
hiFreqList = list(highestFreq["number"])
wordString = ''

for i in range(0,len(hiFreqList)): ## iterate through hiFreqList
    if i != len(hiFreqList)-1 and len(hiFreqList) != 1: ##if not at the end add number to string
        wordString = wordString + str(hiFreqList[i]) + ', '
    elif i == len(hiFreqList)-1 and len(hiFreqList) != 1: #if at the end then print string with and
        wordString = wordString + 'and ' + str(hiFreqList[i])
    elif len(hiFreqList) == 1: # if only one highest frequency number then just print
        wordString = str(hiFreqList[i])

print('Most frequent item in the list is', wordString)


