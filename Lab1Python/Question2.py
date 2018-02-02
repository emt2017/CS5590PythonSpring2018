'''Question 2'''
'''Write a Python function that accepts a sentence of words from user and display the following: '''
#a) Middle word
#b) Longest word in the sentence
#c) Reverse all the words in sentence

'''Import Libraries'''
import string
import sys

'''Function'''
def midLongReverse(string):

#get/store string.split()
    stringArray = string.split()

#a) Middle word
#divide length of array in half
    arrayLength = len(stringArray)/2
#if integer display array[length/2]
    if arrayLength%1==0:
        print('The middle words in the sentence are: ', '[', stringArray[int(arrayLength)-1], '.',stringArray[int(arrayLength)], ']')
#else non integer is decimal then +- 0.5 display array.array
    else: print('The middle words in the sentence are: ','[',stringArray[int(arrayLength)],']')
#b) Longest word in the sentence
#iterate through array compare word.length
    longestWord = ''

    for i in range(0,len(stringArray)):
        if len(stringArray[i]) > len(longestWord):
            longestWord = stringArray[i]

    print('The longest word in the sentence is: ',longestWord)

#c) Reverse all the words in sentence
    reverseWords = string[::-1]
    reverseWords = reverseWords.split()[::-1]
    reverseName = ''
    for i in range(0,len(reverseWords)):
        reverseName = reverseName + reverseWords[i] + ' '
    print('Your sentence in reverse: ',reverseName)

midLongReverse('My name is Jacqueline fernandez Dsouza')