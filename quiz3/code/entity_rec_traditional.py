import nltk
from nltk import word_tokenize,pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
text="Rights of the Accuser .. During the student conduct process, the victim has the following rights: .. • The right to be notified in writing of their rights in the conduct process; .. • The right to be assisted by various campus entities including but not limited to SAVIP , OSCAI, DLES; .. • The right to have an advisor of their choosing accompany them throughout the duration of the conduct process; .. • The right to submit a victim impact statement to the Hearing Officer or Council for consideration; .. • The right to have their past behaviors excluded in a University Conduct Hearing where responsibility is being de-.. termined; .. • The right to accommodations in giving testimony consistent with providing a safe atmosphere, and consistent .. with the rights of the accused; • The right to be notified in writing of the final determination and any sanction imposed on the accused as a result .. of the conduct process; • The right to receive a copy of the formal charges sent to the accused student; .. • The right to be notified of the date, time, and place of hearings at least three University business days prior to the .. hearing; • The right to have the hearing authority consider as an aggravating factor when sanctioning the perpetrator .. whether the perpetrator provided alcohol or other drugs in the commission of a sexual assault; • The right to be notified of the findings and sanctions/outcome of the hearing within a timeframe close to that in .. which the charged student was notified; • The right to appeal the outcome based on a due process error or on information that could not have been avail-.. able at the time of the hearing; and • The right to changes in academic, living, transportation, or working situations to avoid a hostile environment .. page 18 Annual Security and Fire Safety Report"

tokens = word_tokenize(text)
tag=pos_tag(tokens)
print(tag)

ne_tree = nltk.ne_chunk(tag)
print(ne_tree)