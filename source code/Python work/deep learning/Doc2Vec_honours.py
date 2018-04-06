import numpy as np
from gensim import models
import gensim
import smart_open

#define courpus text file
corpusFile = "C:/Users/Norbert/Desktop/NEW_cleanText_conversationBased_Testing_Labelled.txt"

# read corpus of text in a smart and efficient way
def read_corpus(fname):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            index = line.find(' ')
            conversation = str(line[(index + 1):])
            yield gensim.models.doc2vec.TaggedDocument(conversation, [i])

# function to take care of the pre-trained doc2vec model
def DOC2VEC_pretrained(modelPath, outputFile):
	# get the text corpus
    train_corpus = list(read_corpus(corpusFile))
    print "finished assembling the training corpus..."

    #model = models.doc2vec.DocvecsArray(modelSavingPath)

	# load in doc2vec model
    model = models.Doc2Vec.load(modelPath)
    print "model loaded in ..."

    shape = (len(train_corpus), 300)		# define shape of the vectors
    vectors = np.zeros(shape, dtype=float)
    for num in range(0, len(train_corpus)):
        if num % 10000 == 0:
            print "PROGRESS: conversation " + str(num)
        vectors[num] = model.infer_vector(train_corpus[num].words)		# get the document vectors from the model
    print "finished calculating vectors..."

    with open(outputFile, 'w+') as file_vector:					#write all vector space data to csv files, to be analyzed by ML algorithms
        for k in range(0, len(vectors[0])):
            if k == len(vectors[0]) - 1:
                file_vector.write('dim' + str(k + 1) + '\n')		# create a dimensionality counter for the header of the file
            else:
                file_vector.write('dim' + str(k + 1) + ',')

        for i in range(0, len(vectors)):		# for each row
            for j in range(0, len(vectors[i])):		# for each column
                if j == len(vectors[i]) - 1:
                    file_vector.write(str(vectors[i][j]) + '\n')		# write data
                else:
                    file_vector.write(str(vectors[i][j]) + ',')		# write data
    print "finished writing vectors to csv..."

# training your own doc2vec model
def trainingDoc2Vec():
    # import data
    corpusFile = "C:/Users/Norbert/Desktop/NEW_cleanText_conversationBased_Testing_Labelled.txt"
    modelPath = "C:/Users/Norbert/Desktop/cleanText_dim400_doc2Vec.bin"

    model = None
    train_corpus = list(read_corpus(corpusFile))		# get train corpus
    print "finished assembling the training corpus..."
    model = models.doc2vec.Doc2Vec.load(modelPath)		# load in model
    print "model loaded in ..."

    shape = (len(train_corpus), 400)
    vectors = np.zeros(shape, dtype=float)
    for num in range(0, len(train_corpus)):
        vectors[num] = model.docvecs[num]		# get document vectors

    print "finished loading vectors into numpy array..."

    outfiletsv = 'C:/Users/Norbert/Desktop/doc2vec_dim400.csv'
    with open(outfiletsv, 'w+') as file_vector:				#write all vector space data to csv files, to be analyzed by ML algorithms
        for k in range(0, len(vectors[0])):
            if k == len(vectors[0]) - 1:
                file_vector.write('dim' + str(k + 1) + '\n')		# create a dimensionality counter for the header of the file
            else:
                file_vector.write('dim' + str(k + 1) + ',')

        for i in range(0, len(vectors)):		# for each row
            for j in range(0, len(vectors[i])):		# for each column
                if j == len(vectors[i]) - 1:
                    file_vector.write(str(vectors[i][j]) + '\n')		# write data
                else:
                    file_vector.write(str(vectors[i][j]) + ',')		# write data
    print "finished writing vectors to csv..."

#testing code
if __name__ == "__main__":
    # Doc2Vec wiki
    modelSavingPath = "C:/Users/Norbert/Desktop/consulting/Doc2Vec/enwiki_doc2vec.bin"
    outputFile = "C:/Users/Norbert/Desktop/doc2vec/doc2vec_enwiki.csv"
    DOC2VEC_pretrained(modelSavingPath, outputFile)
    print "finished wiki model"

    # Doc2Vec apnews
    modelSavingPath = "C:/Users/Norbert/Desktop/consulting/Doc2Vec/apnews_doc2vec.bin"
    outputFile = "C:/Users/Norbert/Desktop/doc2vec/doc2vec_apnews.csv"
    DOC2VEC_pretrained(modelSavingPath, outputFile)
    print "finished apnews model"