import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

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

# article_text = convertPdf2TxtWithTika('../data/doc.pdf')

article_text = "Rights of the Accuser .. During the student conduct process, the victim has the following rights: .. • The right to be notified in writing of their rights in the conduct process; .. • The right to be assisted by various campus entities including but not limited to SAVIP , OSCAI, DLES; .. • The right to have an advisor of their choosing accompany them throughout the duration of the conduct process; .. • The right to submit a victim impact statement to the Hearing Officer or Council for consideration; .. • The right to have their past behaviors excluded in a University Conduct Hearing where responsibility is being de-.. termined; .. • The right to accommodations in giving testimony consistent with providing a safe atmosphere, and consistent .. with the rights of the accused; • The right to be notified in writing of the final determination and any sanction imposed on the accused as a result .. of the conduct process; • The right to receive a copy of the formal charges sent to the accused student; .. • The right to be notified of the date, time, and place of hearings at least three University business days prior to the .. hearing; • The right to have the hearing authority consider as an aggravating factor when sanctioning the perpetrator .. whether the perpetrator provided alcohol or other drugs in the commission of a sexual assault; • The right to be notified of the findings and sanctions/outcome of the hearing within a timeframe close to that in .. which the charged student was notified; • The right to appeal the outcome based on a due process error or on information that could not have been avail-.. able at the time of the hearing; and • The right to changes in academic, living, transportation, or working situations to avoid a hostile environment .. page 18 Annual Security and Fire Safety Report"

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)
