from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from icecream import ic

def train(
    model: nn.Module,
    training_set: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
    device=torch.device('cpu')
):

    # Set the model to training mode and fix the upper bound on gradient norm
    model.train()
    model.to(device)
    max_grad_norm = 1

    # Obtain the number of training examples
    num_batches = len(training_set)

    for _ in range(num_epochs):

        training_set = tqdm(training_set, desc=f"Epoch: {_ + 1}/{num_epochs}")
        losses = []

        for current_batch, batch in enumerate(training_set):

            #Separate the batch into input_ids and attention_mask
            input_ids, attention_mask = batch['input']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = batch['label']
            labels = labels.to(device)

            # Zero gradients for the model parameters, does the same thing as optimizer.zero_grad()
            model.zero_grad()

            ic(input_ids.size())
            ic(attention_mask.size())
            ic(labels.size())



            # Feed the batch into the model
            output = model(input_ids, attention_mask, labels)

            input_ids.detach()
            attention_mask.detach()
            labels.detach()
            del input_ids
            del attention_mask
            del labels
            gc.collect()
            torch.cuda.empty_cache()

            # Calculate and backpropagate loss, clip gradient norm
            loss = output['loss']

            loss.backward()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update parameters and learning rate
            optimizer.step()

            training_set.set_postfix(
                {"Batch": f"{current_batch}/{num_batches}", "Loss": loss.item()}
            )
            # Clear the GPU memory of batch that we no longer need
            loss.detach()
            del loss
            gc.collect()
            torch.cuda.empty_cache()

            if current_batch % 100 == 0 and current_batch > 0:
                checkpoint_name = f't5_{current_batch}.pt'
                save_dir = '../experiments/' + checkpoint_name
                torch.save(model.state_dict(), save_dir)

        print(f"Mean loss: {sum(losses)/len(losses)}")
        scheduler.step(sum(losses)/len(losses))


    return model




