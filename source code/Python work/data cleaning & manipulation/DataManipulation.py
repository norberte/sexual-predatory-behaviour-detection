import html as parser
from DataCleaning import textCleaning

convID = {}		## conversation id dictionary

## This code splits a very large csv file into smaller chunks 

import sys
import csv
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

labels = []
## open file
with open("C:/Users/Norbert/Desktop/Honours/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem2.txt", "r", encoding="ISO-8859-1") as infile:
    for line in infile:
        temp = line.split('\t')		## tokenize based on tabs
        labels.append(str(temp[0]))
labels = list(set(labels))			## create a unique list (set) of labels
print("finished reading labels ...")

## define input and output files
filePath = "C:/Users/Norbert/Desktop/betterTestingCorpus.csv"
outputPath = "C:/Users/Norbert/Desktop/labelledTestingCorpus.txt"

counter = 0
with open(filePath, "r", encoding="ISO-8859-1") as infile:
    csvfile = csv.reader(infile)		## define csv reader
    for row in csvfile:
        if len(row) < 1:
            continue
        try:
            if row[0] in convID:
                convID[str(row[0])] = convID[str(row[0])] + [str(parser.unescape(row[3]))]	## concatenate to an already existing conversation and attach it's corresponding label
            else:
                convID[str(row[0])] = [str(parser.unescape(row[3]))]		## initialize new conversation with it's label
        except:
            counter+=1

print("finished processing the raw input file ..." + str(counter))

counter = 1
with open(outputPath, 'w', encoding="utf-8") as f:
    for key in convID:
        if key in labels:  # if the conversation id os in the labelled conversation
            text = "__label__predator " + ' '.join(str(x) for x in textCleaning(convID[key]))		## attach predator label to a conversation, after cleaning it
            f.write(str(text) + '\n')
        else:
            text = "__label__non-predator " + ' '.join(str(x) for x in textCleaning(convID[key]))	## attach non-predator label to a conversation, after cleaning it
            f.write(str(text) + '\n')

        if counter % 1000 == 0:
            print("Conversation " + str(counter))		## status update to the console
        counter += 1
