from lxml import etree
import csv

from DataCleaning import preproc
import html as parser

## define input and output files
inf, out = 'C:/Users/Norbert/Desktop/Honours/pan12-sexual-predator-identification-training-corpus-2012-05-01/training-corpus.xml', 'C:/Users/Norbert/Desktop/data/cleanedTrainingCorpus.csv'
context = etree.iterparse(inf, tag='conversation', encoding="ISO-8859-1")			## create an interative parsing tree

i = 0
with open(out, 'w+', encoding="utf-8") as f:
    writer = csv.writer(f)
    for event, elem in context:
        id = elem.xpath('@id')[0]		## get the id from the xml file's xpath
        for child in elem:
            line = child.xpath('@line')		## get the line number from the xml file's xpath
            author = child.xpath('author/text()')		## get the author_id from the xml file's xpath
            text = child.xpath('text/text()')		## get the text content from the xml file's xpath

            line = line[0] if len(line) else ''
            author = author[0] if len(author) else ''
            text = preproc(str(parser.unescape(str(text[0].strip('\n'))))) if len(text) else ''			## pre-process text
            writer.writerow(		# write data for each row
                [
                    id,
                    line,
                    author,
                    text
                ]
            )
            f.flush()			## take care of resetting the output writer
        elem.clear()
        if i % 1000 == 0:
            print(i)		## status update to the console
        i += 1