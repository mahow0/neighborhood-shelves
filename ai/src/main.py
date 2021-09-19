import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from train import train
from evaluate import evaluate
from model import T5Seq2SeqModel
from utils import load_train_test_split, ProductDataset

def main(model, optim_params, train_set, eval_set, num_epochs, save_dir, device = torch.device('cpu')):
    model = T5Seq2SeqModel(model_name = model_name)
    tokenizer = model.tokenizer

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), **optim_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=1,
                                                     verbose=False, threshold=0.001)

    # Train model
    torch.cuda.empty_cache()

    model = train(model, train_set, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs,
                  device=device)

    # Save model weights
    checkpoint_name = f't5_{num_epochs}epochs_{model_name}.pt'
    save_dir = save_dir + checkpoint_name
    torch.save(model.state_dict(), save_dir)

    # Evaluate model
    #acc, precision, recall, f1 = evaluate(model, tokenizer, eval_set, device=device)

    #print(f'Accuracy: {acc}')
    #print(f'Precision: {precision}')
    #print(f'Recall: {recall}')
    #print(f'F1: {f1}')

    #return acc, precision, recall, f1

    pass


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        type = str,
                        help = 'Directory where results (checkpoints + predictions) are saved',
                        default = '../experiments/')

    parser.add_argument('--epochs',
                        type = int,
                        help = 'Number of epochs for training',
                        default = 1)

    parser.add_argument('--train_batch_size',
                        type = int,
                        help = 'Batch size for training',
                        default = 4)

    parser.add_argument('--eval_batch_size',
                        type=int,
                        help='Batch size for eval',
                        default=16)

    parser.add_argument('--lr',
                        type = float,
                        help = 'Learning rate',
                        default = 0.001)

    parser.add_argument('--cuda',
                        type = bool,
                        nargs = '?',
                        default=True)

    # Parse arguments
    args = parser.parse_args()

    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model_name = 'google/t5-v1_1-base'
    model = T5Seq2SeqModel(model_name)

    train_set_params = {'batch_size': args.train_batch_size, 'shuffle': True, 'num_workers': 0}
    eval_set_params = {'batch_size': args.eval_batch_size, 'num_workers': 0}

    #Retrieve datasets
    train_set, eval_set = load_train_test_split('../data/train.csv')

    train_set = DataLoader(ProductDataset(model.tokenizer, train_set), **train_set_params)
    eval_set = DataLoader(ProductDataset(model.tokenizer, eval_set), **eval_set_params)

    optim_params = {'lr': args.lr}
    #Initialize model
    main(model,
         optim_params=optim_params,
         train_set = train_set,
         eval_set = eval_set,
         num_epochs=args.epochs,
         save_dir = args.output_dir,
         device = device)




