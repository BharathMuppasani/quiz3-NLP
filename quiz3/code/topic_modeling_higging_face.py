from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

mname = "cristian-popa/bart-tl-ng"
tokenizer = AutoTokenizer.from_pretrained(mname)
model = AutoModelForSeq2SeqLM.from_pretrained(mname)
#used the summary obtained from huggingface model
input = "Rights of the Accuser .. During the student conduct process, the victim has the following rights: .. The right to be notified in writing of their rights in the conduct process; .. The right to be assisted by various campus entities including but not limited to SAVIP , OSCAI, DLES; .. The right to have an advisor of their choosing accompany them throughout the duration of the conduct process; .. The right to submit a victim impact statement to the Hearing Officer or Council for consideration; .. • The right to have their past behaviors excluded in a University Conduct Hearing where responsibility is being de-.. termined; .. • The right to accommodations in giving testimony consistent with providing a safe atmosphere, and consistent .. with the rights of the accused; • The right to be notified in writing of the final determination and any sanction imposed on the accused as a result .. of the conduct process; • The right to receive a copy of the formal charges sent to the accused student; .. • The right to be notified of the date, time, and place of hearings at least three University business days prior to the .. hearing; • The right to have the hearing authority consider as an aggravating factor when sanctioning the perpetrator .. whether the perpetrator provided alcohol or other drugs in the commission of a sexual assault; • The right to be notified of the findings and sanctions/outcome of the hearing within a timeframe close to that in .. which the charged student was notified; • The right to appeal the outcome based on a due process error or on information that could not have been avail-.. able at the time of the hearing; and • The right to changes in academic, living, transportation, or working situations to avoid a hostile environment .. page 18 Annual Security and Fire Safety Report"
enc = tokenizer(input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
outputs = model.generate(
    input_ids=enc.input_ids,
    attention_mask=enc.attention_mask,
    max_length=15,
    min_length=1,
    do_sample=False,
    num_beams=25,
    length_penalty=1.0,
    repetition_penalty=1.5
)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # windows live messenger