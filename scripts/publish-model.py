from transformers import BartForConditionalGeneration

MODEL_FOLDER = "sciq"
model = BartForConditionalGeneration.from_pretrained(f"./{MODEL_FOLDER}")
model.push_to_hub("nlp-group-6/sciq-question-generator-answer-given", tok)