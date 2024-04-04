import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration


def train_options_generator():
    # make sure to include cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    data = load_dataset("allenai/sciq")

    max_input = 512
    max_target = 128
    batch_size = 36

    args1 = Seq2SeqTrainingArguments(
        output_dir="./results_prediction_question",
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=32,
        predict_with_generate=True,
        eval_accumulation_steps=32,
        fp16=torch.cuda.is_available()  # available only with CUDA
    )

    MODEL_FOLDER = "models/sciq"

    questions_model = BartForConditionalGeneration.from_pretrained(f"./{MODEL_FOLDER}")
    questions_trainer = Seq2SeqTrainer(
        questions_model,
        args1,
        tokenizer=tokenizer,
    )

    def pre_process_data_question_model(data):
        # tokenize the data
        inputs = tokenizer(data['support'], data['correct_answer'], padding="max_length", truncation=True, max_length=max_input, return_tensors="pt")
        targets = tokenizer(data['question'], padding="max_length", truncation=True, max_length=max_target, return_tensors="pt")
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": targets.input_ids}

    def add_generated_question(dataset):
        question_data = dataset.map(pre_process_data_question_model, batched=True)
        predictions = questions_trainer.predict(question_data, max_length=64)
        valid_tokens = []
        for prediction in predictions[0]:
            valid_tokens.append([token for token in prediction if token != -100])
        generated_questions = tokenizer.batch_decode(valid_tokens, skip_special_tokens=True)
        return dataset.add_column("generated_question", generated_questions)

    data['validation'] = add_generated_question(data['validation'])
    data['train'] = add_generated_question(data['train'])
    data['test'] = add_generated_question(data['test'])

    print(data)
    data.save_to_disk("datasets/generated-questions.hf")


if __name__ == "__main__":
    train_options_generator()
