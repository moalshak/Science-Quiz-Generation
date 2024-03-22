import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration


def main():
    # make sure to include cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    data = load_dataset("allenai/sciq")
    train_data = data['train']
    eval_data = data['test']
    test_data = data['validation']

    max_input = 512
    max_target = 128
    batch_size = 8

    # dataset has:
    # question, distractor3, distractor1, distractor2, correct_answer, support
    def pre_process_data(data, tokenizer):
        # tokenize the data
        inputs = tokenizer(data['correct_answer'] + "\n" + data['support'], padding="max_length", truncation=True, max_length=max_input,
                           return_tensors="pt")
        targets = tokenizer(data['question'], padding="max_length", truncation=True, max_length=max_target,
                            return_tensors="pt")
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": targets.input_ids}

    train_data = train_data.map(pre_process_data, batched=True).shuffle(seed=42)
    eval_data = eval_data.map(pre_process_data, batched=True).shuffle(seed=42)
    test_data = test_data.map(pre_process_data, batched=True).shuffle(seed=42)

    model.to(device)
    args = Seq2SeqTrainingArguments(
        output_dir="./results",
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
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    # lets save the model
    OUT_DIR = "sciq"
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == '__main__':
    main()
