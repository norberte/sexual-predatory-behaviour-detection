from DataCleaning import preproc
import html as parser
from itertools import islice


## This code splits a very large csv file into smaller chunks 

import sys
import csv
maxInt = sys.maxsize
decrement = True

## takes care of Python not crashing for using too large integers
while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

labels = {}

## open file
with open("C:/Users/Norbert/Desktop/Honours/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem2.txt", "r", encoding="ISO-8859-1") as infile:
    for line in infile:
        temp = line.split('\t')		## tokenize based on tabs
        if temp[0] in labels:
            labels[str(temp[0])] = labels[str(temp[0])] + [int(temp[1])]		## look at the conversation id and attach it's corresponding label
        else:
            labels[str(temp[0])] = [int(temp[1])]

print("finished reading labels ...")

filePath = "C:/Users/Norbert/Desktop/betterTestCorpus.csv"
outputPath4 = "C:/Users/Norbert/Desktop/labelledLines_TestCorpus4.txt"

## some hard-coded values for processing a specific part of the file
counter = 0
lineCounter = 2999999

with open(filePath, "r", encoding="ISO-8859-1") as infile:
    csvfile = csv.reader(infile)
    for row in islice(csvfile, 3000000, 4190131):
        lineCounter += 1
        if lineCounter % 1000 == 0:			## provide a status update to the console
            print("Conversation line: " + str(lineCounter))
        if len(row) < 1:
            continue

        with open(outputPath4, 'a', encoding="utf-8") as f:
            try:
                if str(row[0]) in labels:  # if the conversation id is in the labelled conversation
                    if int(row[1]) in labels[str(row[0])]:
                        f.write("__label__predator " + str(preproc(str(parser.unescape(row[3])))) + '\n')
                    else:				# if the conversation id is in not the labelled conversation
                        f.write("__label__non-predator " + str(preproc(str(parser.unescape(row[3])))) + '\n')
                else:		
                    f.write("__label__non-predator " + str(preproc(str(parser.unescape(row[3])))) + '\n')
            except:
                print(row)
                counter += 1		# count how many times it fails (if it does)

print("finished processing the raw input file ..." + str(counter))


