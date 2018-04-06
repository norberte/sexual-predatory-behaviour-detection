from autocorrect import spell
from gensim import parsing
from nltk.corpus import stopwords
import re, nltk

# A custom stoplist
STOPLIST = list(stopwords.words('english') + ["n't", "'s", "'m", "ca", "'ve", "'ll", "'d"])		
noNum = re.compile(r'[^a-zA-Z ]')  # number and punctuation remover

mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)		# removes twitter handles
cleanr = re.compile('<.*?>')		# remove punctuation

def __whiteSpaceAndNumericRemoval(text):
    cleanedText = parsing.preprocessing.strip_multiple_whitespaces(text)			# remove multiple white spaces
    #cleanedText = parsing.preprocessing.strip_numeric(cleanedText)		# remove numeric values
    #cleanedText = parsing.preprocessing.strip_tags(cleanedText)		# remove any kind of tags

    # get rid of newlines
    #cleanedText = cleanedText.strip('\n')

    # replace twitter @mentions
    # cleanedText = mentionFinder.sub("@MENTION", cleanedText)

    # get rid of html links and tags
    cleanedText = re.sub(cleanr, '', cleanedText)

    # replace HTML symbols
    # cleanedText = cleanedText.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    return cleanedText
	
# spell-checker
def __autoCorrect(s):
    return str(spell(s))

# function that cleans the text
def __clean(text):
    clean_text = noNum.sub(' ', text)
    tokens = nltk.word_tokenize(clean_text)
    newTokens = []
    for token in tokens:
        if len(token) < 20:
            newTokens.append(token)
    filtered_words = [w for w in newTokens if not w in STOPLIST]  # filter using the stoplist of words or abbreviations
    return filtered_words
	
# adds spaces after each word went through pre-processing
def __spaces(s):
    return ' '.join(s.split())

# function that wraps all the pre-processing together
def preproc(s):
    text = __whiteSpaceAndNumericRemoval( s.lower())
    listOfWords = __clean(text)
    #words = []
    #for word in listOfWords:
    #    words.append(__autoCorrect(word))
    return ' '.join(str(x) for x in listOfWords)

if __name__ == '__main__':
	# just for testing purposes
    conversationDataFile = "C:/Users/Norbert/Desktop/data/conversationBased_Testing_Labelled_new.txt"  # read data file
    outputFile = 'C:/Users/Norbert/Desktop/TWE output/cleanText_conversationBased_Testing_Labelled.txt' # output file
    counter = 0
    label = cleanConversation = ""
    with open(outputFile, 'w+') as file_vector:
        with open(conversationDataFile, "r") as dataFile:
            for line in dataFile:
                print "PROGRESS: conversation " + str(counter)
                counter += 1
                line = line.strip('\n')
                index = line.find(' ')
                conversation = str(line[(index + 1):])
                if conversation.isspace():
                    continue
                else:
                    label = str(line[0:index])                 # predator/ non-predator label
                    cleanConversation = preproc(conversation)    #pre-process the conversation
                    file_vector.write(label + " " + cleanConversation + '\n')

