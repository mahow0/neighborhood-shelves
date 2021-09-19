from transformers import Trainer, TrainingArguments


def train(model , training_args : TrainingArguments, train_set, eval_set):

    trainer = Trainer(model = model,
                      args = training_args,
                      train_dataset = train_set,
                      eval_dataset = eval_set
    )

    trainer.train()

    return trainer


