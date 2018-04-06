import numpy as np
import nltk
from itertools import islice
import sys
reload(sys)
import smart_open
sys.setdefaultencoding('utf-8')
from gensim.models import KeyedVectors

# load in pre-trained model
model = KeyedVectors.load_word2vec_format('C:/Users/Norbert/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
print "model loaded in..."

conversationDataFile = "C:/Users/Norbert/Desktop/NEW_cleanText_conversationBased_Testing_Labelled.txt"    #read data file
minDoc = []			#create list for min, max and minMax word2vec model's vector outputs
maxDoc = []
minMaxDoc = []

value = 0
with smart_open.smart_open(conversationDataFile, encoding="utf-8") as f:			# read conversation files
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print "----------------------PROGRESS: conversation " + str(i) + "----------------------------"

        line = line.strip('\n')		# strip conversations into each line
        index = line.find(' ')		# find the first space and put an index to it
        conversation = str(line[(index + 1):])		# find and extract the conversation text
        if conversation.isspace():
            continue		# check if it's not an empty conversation
        else:
            tokens = []
            try:
                tokens = nltk.word_tokenize(conversation)   # tokenize conversation
            except:
                tokens = nltk.word_tokenize(str(conversation.encode('utf-8')))		# tokenize with utf-8 encoding for odd characters
            conversationVectors = []
            for i in range(0, len(tokens)):
                try:
                    conversationVectors.append(model[str(tokens[i])])		# append the conversation lines together
                except:
                    value += 1

            if len(conversationVectors) > 0:			#do some word vector aggregation using the min, max and minMax (concatenated min and max) functions
                a = np.array(conversationVectors)
                maxVec = a.max(axis=0)
                maxDoc.append(maxVec)
                minVec = a.min(axis=0)
                minDoc.append(minVec)
                minMax200 = np.concatenate([minVec,maxVec])
                minMaxDoc.append(minMax200)
            else:
                maxDoc.append([0] * 300)			# create zero vectors if something went wrong
                minDoc.append([0] * 300)
                minMaxDoc.append([0] * 600)

print "finished calculating vectors"

minDoc = np.array(minDoc)			#convert list of vectors into numpy arrays for min, max and minMax pre-trained word2vec model's vector outputs
maxDoc = np.array(maxDoc)
minMaxDoc = np.array(minMaxDoc)

print "finished converting lists into numpy arrays"

outfiletsv = 'C:/Users/Norbert/Desktop/Word2Vec/minDoc_word2vec_pretrained.csv'		
with open(outfiletsv, 'w+') as file_vector:		#write all vector space data to csv files, to be analyzed by ML algorithms
    for k in range(0, len(minDoc[0])):
        if k == len(minDoc[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')		# create a dimensionality counter for the header of the file
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minDoc)):		# for each row
        for j in range(0, len(minDoc[i])):		# for each data
            if j == len(minDoc[i]) - 1:
                file_vector.write(str(minDoc[i][j]) + '\n')			# write data
            else:
                file_vector.write(str(minDoc[i][j]) + ',')			# write data

print "finished writing minDoc to file"

outfiletsv = 'C:/Users/Norbert/Desktop/Word2Vec/maxDoc_word2vec_pretrained.csv'				#write all vector space data to csv files, to be analyzed by ML algorithms
with open(outfiletsv, 'w+') as file_vector:
    for k in range(0, len(maxDoc[0])):
        if k == len(maxDoc[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')			# create a dimensionality counter for the header of the file
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(maxDoc)):				# for each row
        for j in range(0, len(maxDoc[i])):			# for each column	
            if j == len(maxDoc[i]) - 1:
                file_vector.write(str(maxDoc[i][j]) + '\n')		# write data
            else:
                file_vector.write(str(maxDoc[i][j]) + ',')			# write data	

print "finished writing maxDoc to file"

outfiletsv = 'C:/Users/Norbert/Desktop/Word2Vec/minMaxDoc_word2vec_pretrained.csv'
with open(outfiletsv, 'w+') as file_vector:					#write all vector space data to csv files, to be analyzed by ML algorithms
    for k in range(0, len(minMaxDoc[0])):
        if k == len(minMaxDoc[0]) - 1:
            file_vector.write('dim' + str(k + 1) + '\n')			# create a dimensionality counter for the header of the file
        else:
            file_vector.write('dim' + str(k + 1) + ',')

    for i in range(0, len(minMaxDoc)):		# for each row
        for j in range(0, len(minMaxDoc[i])):		#for each column
            if j == len(minMaxDoc[i]) - 1:
                file_vector.write(str(minMaxDoc[i][j]) + '\n')		# write data
            else:
                file_vector.write(str(minMaxDoc[i][j]) + ',')		# write data

print "finished writing minMaxDoc to file"




