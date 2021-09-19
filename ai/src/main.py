import argparse
from transformers import TrainingArguments
from train import train
from evaluate import evaluate
from model import Seq2SeqModel

def main(model, training_args, train_set, eval_set):

    trainer = train(model = model,
                    training_args = training_args,
                    train_set = train_set,
                    eval_set = eval_set)

    evaluate(trainer)












if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        type = str,
                        help = 'Directory where results (checkpoints + predictions) are saved',
                        default = '../experiments/')

    parser.add_argument('--epochs',
                        type = int,
                        help = 'Number of epochs for training')

    parser.add_argument('--train_batch_size',
                        type = int,
                        help = 'Batch size for training',
                        default = 32)

    parser.add_argument('--eval_batch_size',
                        type=int,
                        help='Batch size for eval',
                        default=64)

    parser.add_argument('--lr',
                        type = float,
                        help = 'Learning rate',
                        default = 0.001)

    # Parse arguments
    args = parser.parse_args()

    # Training hyperparameters and configuration
    training_kwargs = { 'output_dir' : args.output_dir,
               'num_train_epochs' : args.epochs,
               'per_device_train_batch_size' : args.train_batch_size,
               'per_device_eval_batch_size' : args.eval_batch_size,
               'learning_rate' : args.lr,
               'logging_strategy' : 'epoch',
               'lr_scheduler_type' : 'cosine',
               'warmup_steps' : 0
    }

    training_args = TrainingArguments(**training_kwargs)

    #Retrieve datasets
    train_set = None
    eval_set = None

    #Initialize model
    model_name = ''
    model = Seq

    main(model)




