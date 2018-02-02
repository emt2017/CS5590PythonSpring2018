'''Question 3'''
'''given a list of n numbers, write a Python program to find triplets in the list which gives the sum of zero'''
# input => [1, -7, 6, 2, -1, 2, 0, -2, 0]
# output => [(7,-1,-2)]

def tripletSums(numArray):
#for input compare all
    for i in range(0, len(numArray)-2):
#for input2 = input+1 compare all
        for j in range(i+1, len(numArray)-1):
#for input3 = input2+1 compare all
            for k in range(j+1, len(numArray)):
#add all numbers if = 0
                if numArray[i]+numArray[j]+numArray[k] == 0:
                    storeArray = [numArray[i],numArray[j],numArray[k]]
# print combination of numbers
                    print(storeArray)


numArray = [1, -7, 6, 2, -1, 2, 0, -2, 0]

tripletSums(numArray)