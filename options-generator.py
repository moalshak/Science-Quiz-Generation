import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration


def train_options_generator():
    # make sure to include cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    data = load_dataset("allenai/sciq")
    train_data = data['train']
    val_data = data['validation']

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
        fp16=True  # available only with CUDA
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

    val_data = val_data.select(range(30))
    question_val_data = val_data.map(pre_process_data_question_model, batched=True)
    val_predictions = questions_trainer.predict(question_val_data, max_length=64)
    for idx in range(len(val_predictions)):
        val_predictions[0][idx] = [token for token in val_predictions[0][idx] if token != -100]
    val_generated_questions = tokenizer.batch_decode(val_predictions[0], skip_special_tokens=True)
    val_data = val_data.add_column("generated_question", val_generated_questions)

    train_data = train_data.select(range(30))
    question_train_data = train_data.map(pre_process_data_question_model, batched=True)
    train_predictions = questions_trainer.predict(question_train_data, max_length=64)
    for idx in range(len(train_predictions)):
        train_predictions[0][idx] = [token for token in train_predictions[0][idx] if token != -100]
    train_generated_questions = tokenizer.batch_decode(train_predictions[0], skip_special_tokens=True)
    train_data = train_data.add_column('generated_question', train_generated_questions)

    # dataset has:
    # question, distractor3, distractor1, distractor2, correct_answer, support
    def pre_process_data(data):
        question_answer_context = [question + "</s><s>" + correct_answer + "</s><s>" + support for
                                   question, correct_answer, support in
                                   zip(data['generated_question'], data['correct_answer'], data['support'])]

        # tokenize the data
        inputs = tokenizer(question_answer_context, padding="max_length", truncation=True, max_length=max_input,
                           return_tensors="pt")
        targets = tokenizer(data['distractor1'], data['distractor2'], data['distractor3'], padding="max_length",
                            truncation=True, max_length=max_target, return_tensors="pt")
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": targets.input_ids}

    train_data = train_data.map(pre_process_data, batched=True).shuffle(seed=42)
    val_data = val_data.map(pre_process_data, batched=True).shuffle(seed=42)

    # print(tokenizer.decode(train_data['input_ids'][1], skip_special_tokens=False))

    model.to(device)

    args = Seq2SeqTrainingArguments(
        output_dir="./results_option_generation",
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
        fp16=True  # available only with CUDA
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    # lets save the model
    OUT_DIR = "models/sciq_options_generator"
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    train_options_generator()
