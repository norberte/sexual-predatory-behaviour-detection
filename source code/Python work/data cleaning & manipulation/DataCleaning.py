import subprocess
import re
from autocorrect import spell
from gensim import parsing

mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
cleanr = re.compile('<.*?>')		

def textCleaning(messages):
    cleanLines = []				# stores a list of clean lines
    for line in messages:
        cleanLines.append(preproc(line))		# joins each line in the conversations

    return cleanLines

# function that detects phrases using bigrams
def __phraseDetection(text, bigram):
    bigrams = bigram[list(text.split())]		# apply bigram model
    bigrams_str = ' '.join(str(x) for x in bigrams)
    return bigrams_str

def __whiteSpaceAndNumericRemoval(text):
    cleanedText = parsing.preprocessing.strip_multiple_whitespaces(text)		# remove multiple white spaces
    cleanedText = parsing.preprocessing.strip_numeric(cleanedText)	# remove numeric values
    cleanedText = parsing.preprocessing.strip_tags(cleanedText)		# remove any kind of tags

    # get rid of newlines
    cleanedText = cleanedText.strip('\n')

    # replace twitter @mentions
    cleanedText = mentionFinder.sub("@MENTION", cleanedText)

    # get rid of html links
    cleanedText = re.sub(cleanr, '', cleanedText)

    # replace HTML symbols
    cleanedText = cleanedText.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    return cleanedText

# spell-checker
def __autoCorrect(s):
    return str(spell(s))

# adds spaces after each word went through pre-processing
def __spaces(s):
    return ' '.join(s.split())

# function that wraps all the pre-processing together
def preproc(s):
    return __spaces(__whiteSpaceAndNumericRemoval(s.lower() ) )