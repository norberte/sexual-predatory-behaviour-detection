import sys
import csv


## This code splits a very large csv file into smaller chunks 

maxInt = sys.maxsize
decrement = True
convID = {}

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

# define input and output conversations data files		
labels = []
#inputFile = "C:/Users/Norbert/Desktop/Honours/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem1.txt"
inputFile = "C:/Users/Norbert/Desktop/Honours/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem2.txt"
with open(inputFile, "r", encoding="utf-8") as infile:
    for line in infile:
        temp = line.split('\t')
        labels.append(str(temp[0]))
labels = list(set(labels))
print("finished reading labels ...")

# define input and output label files
filePath = "C:/Users/Norbert/Desktop/data/cleanedTestingCorpus.csv"
#outputPath = "C:/Users/Norbert/Desktop/data/conversationBased_Testing_Labelled_new.txt"
outputPath = "C:/Users/Norbert/Desktop/conversationBased_Testing_Labelled_new.csv"

issueCounter = 0
lineCounter = 1
## open file
with open(filePath, "r", encoding="utf-8") as infile:
    csvfile = csv.reader(infile)		## define csv reader
    for row in csvfile:
        if lineCounter % 50000 == 0:
            print("Reading line " + str(lineCounter))		# status update when reading every 50000 line
        lineCounter += 1
        if len(row) < 1:
            continue
        try:
            if row[0] in convID:
                convID[str(row[0])] = convID[str(row[0])] + [str(row[3])]    ## concatenate to an already existing conversation and attach it's corresponding conversation level label
            else:
                convID[str(row[0])] = [str(row[3])]				## initialize new conversation with it's conversation level label
        except:
            issueCounter+=1

print("finished processing the raw input file ..." + str(issueCounter))

counter = 1
with open(outputPath, 'w', encoding="utf-8") as f:
    for key in convID:
        if key in labels:  # if the conversation id is in the labelled conversation
            f.write(' '.join(str(x) for x in convID[key]) + '\n')

        if counter % 50000 == 0:
            print("Conversation " + str(counter))			# status update when writing every 50000 output line
        counter += 1
