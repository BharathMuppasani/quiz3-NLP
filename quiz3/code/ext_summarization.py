import math
import tika
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

from tika import parser
def convertPdf2TxtWithTika(in_pdf_file):
    # Load a file and extract information
    print ("INFO: - reading file = " + in_pdf_file)
    
    raw = parser.from_file(in_pdf_file)
    text = raw['content']
    
    ## Post-processing explained at: 
    # https://medium.com/@justinboylantoomey/fast-text-extraction-with-python-and-tika-41ac34b0fe61
    # Convert to string
    text = str(text)
    # Ensure text is utf-8 formatted
    safe_text = text.encode('utf-8', errors='ignore')
    # Escape any \ issues
    safe_text = str(safe_text).replace('\\', '\\\\').replace('"', '\\"')
    
    return text

# text_str = convertPdf2TxtWithTika('../data/doc.pdf')
text_str = "Rights of the Accuser .. During the student conduct process, the victim has the following rights: .. • The right to be notified in writing of their rights in the conduct process; .. • The right to be assisted by various campus entities including but not limited to SAVIP , OSCAI, DLES; .. • The right to have an advisor of their choosing accompany them throughout the duration of the conduct process; .. • The right to submit a victim impact statement to the Hearing Officer or Council for consideration; .. • The right to have their past behaviors excluded in a University Conduct Hearing where responsibility is being de-.. termined; .. • The right to accommodations in giving testimony consistent with providing a safe atmosphere, and consistent .. with the rights of the accused; • The right to be notified in writing of the final determination and any sanction imposed on the accused as a result .. of the conduct process; • The right to receive a copy of the formal charges sent to the accused student; .. • The right to be notified of the date, time, and place of hearings at least three University business days prior to the .. hearing; • The right to have the hearing authority consider as an aggravating factor when sanctioning the perpetrator .. whether the perpetrator provided alcohol or other drugs in the commission of a sexual assault; • The right to be notified of the findings and sanctions/outcome of the hearing within a timeframe close to that in .. which the charged student was notified; • The right to appeal the outcome based on a due process error or on information that could not have been avail-.. able at the time of the hearing; and • The right to changes in academic, living, transportation, or working situations to avoid a hostile environment .. page 18 Annual Security and Fire Safety Report"


def _create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()): 
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def _find_average_score(sentenceValue) -> int:

    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):

    sentences = sent_tokenize(text)
    total_documents = len(sentences)

    freq_matrix = _create_frequency_matrix(sentences)

    tf_matrix = _create_tf_matrix(freq_matrix)

    count_doc_per_words = _create_documents_per_words(freq_matrix)

    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)

    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)

    sentence_scores = _score_sentences(tf_idf_matrix)

    threshold = _find_average_score(sentence_scores)

    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary


result = run_summarization(text_str)
print(result)