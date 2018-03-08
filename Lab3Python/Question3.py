#3)Write a program  Take an Inputfile.
# Use the simple approach below to summarize a text file:
# -Read the file ################################################################################################
# -Using Lemmatization, apply lemmatization on the words ########################################################
# -Apply the bigram on the text #################################################################################
# -Calculate the word frequency (bi-gram frequency) of the words(bi-grams) ######################################
# -Choose top five bi-grams that has been repeated most #########################################################
# -Go through the original text that you had in the file ########################################################
# -Find all the sentences with those most repeated bi-grams #####################################################
# -Extract those sentences and concatenate ######################################################################
# -Enjoy the summarization

from nltk.stem import WordNetLemmatizer
from nltk.util import bigrams
from nltk.tokenize import sent_tokenize, word_tokenize

with open('readFile', 'r') as readFile: #open file we want to read
    text = readFile.read() #save text from file
    #text = text.split() #split text into a list of words
    #http://www.nltk.org/api/nltk.stem.html
    textSentence = sent_tokenize(text) #make tokenize sentence
    textWord = word_tokenize(text) #make tokenize words
    Lemmatize = WordNetLemmatizer() #prepare lemmatization
    lemmatizedText = [] #initialize new text list for lemmatized words

    #lemmatize words and append to new list
    for word in textWord:
        lemmatizedText.append(Lemmatize.lemmatize(word))

    #https://www.nltk.org/api/nltk.html
    lemTextBiGram = list(bigrams(lemmatizedText))

    bigramFreq = dict() #create a dicitonary to store a frequency next to a number
    for bigram in lemTextBiGram:  #run through dictionary
        if bigram not in bigramFreq: #check if bigram is already in dictionary
            bigramFreq.update({bigram:1}) #if it is not then add to dicitonary
        elif bigram in bigramFreq: #else if bigram in the dictionary
            # then add one to the frequency instead of adding it to the dictionary
            bigramFreq[bigram] = bigramFreq[bigram] + 1

    #https://www.pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/
    sortedBigramFreq=sorted(bigramFreq, reverse = True, key = bigramFreq.__getitem__) #sort the dicitonary
    top5 = sortedBigramFreq[0:5] #get the top five bigrams


    #iterate through sentences
    #check for top 5 bigrams
    #if true append sentence
    #check for duplicate sentences/make duplicates impossible by deleting if found
    bigramSentences = [] #initialize list
    top5String = [] #initialize list

    #turn top 5 bigrams into strings instead of tuples
    for bigram in top5: #run through top 5 bigrams
        for sentence in textSentence: #run through original tokenized sentences
            newSentence = list(bigrams(word_tokenize(sentence))) #make a list of bigrams in the particular sentence
            if bigram in newSentence: #check if bigram is in the that sentence
                bigramSentences.append(sentence) #append to sentence if bigram is in sentence
                textSentence.remove(sentence) #remove to avoid duplicates

    #concatenate the list of sentences
    concatenate = " ".join(bigramSentences)

    print(bigramSentences)
    print(concatenate)
