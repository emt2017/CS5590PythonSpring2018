'''
Consider a shop UMKC with dictionary of all book items with their prices.
Write a program to find the books from the dictionary in the range given by user.
Sample Input: {“python”:50,”web”:3   0,”c”:20,”java”:40}
For range 30 to 40
Sample Output: You can purchase books (web, java)
'''


def bookPriceRng(dictionary, low, high):
    #initialize student ranged value list
    studentPref = []
    #for loop to search through the dictionary given
    for key in dictionary:
        #if cost is more than or less than low and high then store book in student preferences
        if(dictionary[key]>=low and dictionary[key]<=high):
            studentPref.append(key)
    #print the keys the student wants
    print(studentPref)



UMKCLibrary = {"python": 50, "web": 30, "c": 20, "java": 40}

bookPriceRng(UMKCLibrary, 30, 40)