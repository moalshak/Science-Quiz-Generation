import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration


def train_options_generator():
    # make sure to include cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = load_from_disk("datasets/generated-questions.hf")
    train_data = data['train']
    val_data = data['validation']
    print(data)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    max_input = 512
    max_target = 128
    batch_size = 36

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
        fp16=torch.cuda.is_available()  # available only with CUDA
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
