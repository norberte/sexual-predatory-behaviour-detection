import numpy as np
import nltk
from itertools import islice
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

dim200 = "C:/Users/Norbert/Desktop/TWE output/wordVector_dim200.txt"			# define 200 dim wordVector imput files
dim400 = "C:/Users/Norbert/Desktop/TWE output/wordVector_dim400.txt"			# define 400 dim wordVector imput files
dim500 = "C:/Users/Norbert/Desktop/TWE output/wordVector_dim500.txt"			# define 500 dim wordVector imput files

#convert word vectors into a dictionary, where each word's vector represenation can be looked up easily
vec200 = {}
with open(dim200, "r") as infile:
    for line in infile:
        line = line.strip('\n')
        tokens = line.split(' ')
        vec200[str(tokens[0])] = np.array(tokens[1:], dtype=float)			# store vector represenation of a word in a numpy array
print "finished reading dim200 model"

vec400 = {}
with open(dim400, "r") as infile:
    for line in infile:
        line = line.strip('\n')
        tokens = line.split(' ')
        vec400[str(tokens[0])] = np.array(tokens[1:], dtype=float)			# store vector represenation of a word in a numpy array
print "finished reading dim400 model"

vec500 = {}
with open(dim500, "r") as infile:
    for line in infile:
        line = line.strip('\n')
        tokens = line.split(' ')
        vec500[str(tokens[0])] = np.array(tokens[1:], dtype=float)			# store vector represenation of a word in a numpy array
print "finished reading dim500 model"

conversationDataFile = "C:/Users/Norbert/Desktop/data/conversationBased_Testing_Labelled_new.txt"    #read data file
label = ""
# define a bunch of lists to store the resulting transformations on the data
listOfLabels = []
minDoc_dim200 = []
maxDoc_dim200 = []
minMaxDoc_dim200 = []
minDoc_dim400 = []
maxDoc_dim400 = []
minMaxDoc_dim400 = []
minDoc_dim500 = []
maxDoc_dim500 = []
minMaxDoc_dim500 = []

value = 0
counter = 0
with open(conversationDataFile, "r") as dataFile:		# open the conversation file
    for line in dataFile:
        counter += 1
        if counter % 1000 == 0:
            print "----------------------PROGRESS: conversation " + str(counter) + "----------------------------"

        line = line.strip('\n')		# read a conversation and break it into linees
        index = line.find(' ')		# find the spaces, keep and index to the first space
        conversation = str(line[(index + 1):])				# find the conversation's label
        if conversation.isspace():
            continue			# skip, if this conversation is empty spaces
        else:
            tokens = []
            try:
                tokens = nltk.word_tokenize(conversation)   # tokenize conversation
            except:
                tokens = nltk.word_tokenize(str(conversation.encode('utf-8')))
            #tokens = conversation.split(' ')       # conversation words
            label = str(line[0:index])            # predator/ non-predator label
            if label == "__label__non-predator":
                listOfLabels.append(int(0))
            elif label == "__label__predator":
                listOfLabels.append(int(1))

            conversationVectors200 = []			
            conversationVectors400 = []
            conversationVectors500 = []
            for i in range(0, len(tokens)):		# get the different dimensional conversation vectors
                try:
                    conversationVectors200.append(vec200[str(tokens[i])])
                except:
                    value += 1
                try:
                    conversationVectors400.append(vec400[str(tokens[i])])
                except:
                    value += 1
                try:
                    conversationVectors500.append(vec500[str(tokens[i])])
                except:
                    value += 1

            if len(conversationVectors200) > 0:						# do some word vector aggregation using the min, max and minMax (concatenated min and max) functions
                a = np.array(conversationVectors200)
                maxVec200 = a.max(axis=0)
                maxDoc_dim200.append(maxVec200)
                minVec200 = a.min(axis=0)
                minDoc_dim200.append(minVec200)
                minMax200 = np.concatenate([minVec200,maxVec200])
                minMaxDoc_dim200.append(minMax200)
            else:
                maxDoc_dim200.append([0] * 200)					# create zero vectors if something goes wrong
                minDoc_dim200.append([0] * 200)
                minMaxDoc_dim200.append([0] * 400)

            if len(conversationVectors400) > 0:					# do some word vector aggregation using the min, max and minMax (concatenated min and max) functions
                c = np.array(conversationVectors400)
                maxVec400 = c.max(axis=0)
                maxDoc_dim400.append(maxVec400)
                minVec400 = c.min(axis=0)
                minDoc_dim400.append(minVec400)
                minMax400 = np.concatenate([minVec400, maxVec400])
                minMaxDoc_dim400.append(minMax400)
            else:
                maxDoc_dim400.append([0] * 400)					# create zero vectors if something goes wrong
                minDoc_dim400.append([0] * 400)
                minMaxDoc_dim400.append([0] * 800)

            if len(conversationVectors500) > 0:					# do some word vector aggregation using the min, max and minMax (concatenated min and max) functions
                d = np.array(conversationVectors500)
                maxVec500 = d.max(axis=0)
                maxDoc_dim500.append(maxVec500)
                minVec500 = d.min(axis=0)
                minDoc_dim500.append(minVec500)
                minMax500 = np.concatenate([minVec500, maxVec500])
                minMaxDoc_dim500.append(minMax500)
            else:												# create zero vectors if something goes wrong
                maxDoc_dim500.append([0] * 500)
                minDoc_dim500.append([0] * 500)
                minMaxDoc_dim500.append([0] * 1000)

minDoc_dim200 = np.array(minDoc_dim200)			# create numpy arrays from the aggregated conversation vectors
maxDoc_dim200 = np.array(maxDoc_dim200)
minMaxDoc_dim200 = np.array(minMaxDoc_dim200)

minDoc_dim400 = np.array(minDoc_dim400)			# create numpy arrays from the aggregated conversation vectors
maxDoc_dim400 = np.array(maxDoc_dim400)
minMaxDoc_dim400 = np.array(minMaxDoc_dim400)

minDoc_dim500 = np.array(minDoc_dim500)			# create numpy arrays from the aggregated conversation vectors
maxDoc_dim500 = np.array(maxDoc_dim500)
minMaxDoc_dim500 = np.array(minMaxDoc_dim500)

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/minDoc_dim200.csv'			#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(minDoc_dim200[0])):
        if k == len(minDoc_dim200[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')				# create a dimensionality counter for the header of the file
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minDoc_dim200)):						# for each row
        for j in range(0, len(minDoc_dim200[i])):			# for each column
            if j == len(minDoc_dim200[i]) - 1:
                file_vector.write(str(minDoc_dim200[i][j]) + '\n')
            else:
                file_vector.write(str(minDoc_dim200[i][j]) + ',')

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/maxDoc_dim200.csv'			#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(maxDoc_dim200[0])):
        if k == len(maxDoc_dim200[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(maxDoc_dim200)):
        for j in range(0, len(maxDoc_dim200[i])):
            if j == len(maxDoc_dim200[i]) - 1:
                file_vector.write(str(maxDoc_dim200[i][j]) + '\n')
            else:
                file_vector.write(str(maxDoc_dim200[i][j]) + ',')

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim200.csv'		#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(minMaxDoc_dim200[0])):
        if k == len(minMaxDoc_dim200[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minMaxDoc_dim200)):
        for j in range(0, len(minMaxDoc_dim200[i])):
            if j == len(minMaxDoc_dim200[i]) - 1:
                file_vector.write(str(minMaxDoc_dim200[i][j]) + '\n')
            else:
                file_vector.write(str(minMaxDoc_dim200[i][j]) + ',')

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/minDoc_dim400.csv'		#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(minDoc_dim400[0])):
        if k == len(minDoc_dim400[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minDoc_dim400)):
        for j in range(0, len(minDoc_dim400[i])):
            if j == len(minDoc_dim400[i]) - 1:
                file_vector.write(str(minDoc_dim400[i][j]) + '\n')
            else:
                file_vector.write(str(minDoc_dim400[i][j]) + ',')

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/maxDoc_dim400.csv'		#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(maxDoc_dim400[0])):
        if k == len(maxDoc_dim400[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(maxDoc_dim400)):
        for j in range(0, len(maxDoc_dim400[i])):
            if j == len(maxDoc_dim400[i]) - 1:
                file_vector.write(str(maxDoc_dim400[i][j]) + '\n')
            else:
                file_vector.write(str(maxDoc_dim400[i][j]) + ',')


outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim400.csv'			#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(minMaxDoc_dim400[0])):
        if k == len(minMaxDoc_dim400[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minMaxDoc_dim400)):
        for j in range(0, len(minMaxDoc_dim400[i])):
            if j == len(minMaxDoc_dim400[i]) - 1:
                file_vector.write(str(minMaxDoc_dim400[i][j]) + '\n')
            else:
                file_vector.write(str(minMaxDoc_dim400[i][j]) + ',')


outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/minDoc_dim500.csv'		#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(minDoc_dim500[0])):
        if k == len(minDoc_dim500[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minDoc_dim500)):
        for j in range(0, len(minDoc_dim500[i])):
            if j == len(minDoc_dim500[i]) - 1:
                file_vector.write(str(minDoc_dim500[i][j]) + '\n')
            else:
                file_vector.write(str(minDoc_dim500[i][j]) + ',')

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/maxDoc_dim500.csv'		#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(maxDoc_dim500[0])):
        if k == len(maxDoc_dim500[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(maxDoc_dim500)):
        for j in range(0, len(maxDoc_dim500[i])):
            if j == len(maxDoc_dim500[i]) - 1:
                file_vector.write(str(maxDoc_dim500[i][j]) + '\n')
            else:
                file_vector.write(str(maxDoc_dim500[i][j]) + ',')

outfiletsv = 'C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim500.csv'		#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(minMaxDoc_dim500[0])):
        if k == len(minMaxDoc_dim500[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minMaxDoc_dim500)):
        for j in range(0, len(minMaxDoc_dim500[i])):
            if j == len(minMaxDoc_dim500[i]) - 1:
                file_vector.write(str(minMaxDoc_dim500[i][j]) + '\n')
            else:
                file_vector.write(str(minMaxDoc_dim500[i][j]) + ',')


outputFile = 'C:/Users/Norbert/Desktop/TWE output/new_labels.csv'		#write labels to a csv file
with open(outputFile, 'w+') as file_vector:
    file_vector.write("Classification" + '\n')
    for label in listOfLabels:
        file_vector.write(str(label) + '\n')




