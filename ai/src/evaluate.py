from transformers import Trainer


def evaluate(trainer : Trainer):
    trainer.evaluate()


