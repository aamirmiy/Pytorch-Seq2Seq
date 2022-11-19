import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from model import Encoder, Decoder, EncoderDecoder
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from random import random
from attention import AttEncoder, AttDecoder, AttEncoderDecoder
from utils import *

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    train_data, val_data =  read_episodes(args.in_data_fn)
    vocab_to_index, index_to_vocab, len_cut_off = build_tokenizer_table(train_data, vocab_size = 3000)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets, max_output_len = build_output_tables(train_data)
    train_x, train_y = flatten_episodes(train_data)
    val_x, val_y = flatten_episodes(val_data)
    train_set = concat_list(train_x)
    val_set = concat_list(val_x)
    x_train, y_train = encode_data(train_set, train_y, vocab_to_index, targets_to_index, actions_to_index)
    x_val, y_val= encode_data(val_set, val_y, vocab_to_index, targets_to_index, actions_to_index)

    train_dataset = CustomDataset([torch.from_numpy(np.array(xi)) for xi in x_train],[torch.from_numpy(np.array(yi)) for yi in y_train])
    train_loader = DataLoader(dataset = train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate,drop_last=True)

    val_dataset = CustomDataset([torch.from_numpy(np.array(xi)) for xi in x_val],[torch.from_numpy(np.array(yi)) for yi in y_val])
    val_loader = DataLoader(dataset = val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate,drop_last=True)
    return train_loader, val_loader, (vocab_to_index, actions_to_index, index_to_actions, targets_to_index, index_to_targets),max_output_len


def setup_model(args,device, vocab_size, output_size1, output_size2,):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #

    #Vanilla Model
    # enc = Encoder(device, vocab_size, args.emb_dim, args.lstm_dim, args.num_layers)
    # dec = Decoder(device, output_size1, output_size2, args.emb_dim, args.lstm_dim, args.num_layers)
    # model = EncoderDecoder(device,enc,dec,args.lstm_dim,output_size1,output_size2)
    #Attention Model
    enc = AttEncoder(device, vocab_size, args.emb_dim, args.lstm_dim, args.num_layers)
    dec = AttDecoder(device, output_size1, output_size2, args.emb_dim, args.lstm_dim, args.num_layers)
    model = AttEncoderDecoder(device,enc,dec,args.lstm_dim,output_size1,output_size2)
    
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    target_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.
    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_pm_acc = 0.0

    #teacher_forcing = training and random()<0.5

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, input_lens, labels, labels_lens) in loader:
        # put model inputs to device
        teacher_forcing = training and random()<0.5
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        loss=0.0
        if teacher_forcing:
            act,tar = model(inputs, input_lens,labels, labels_lens)
            
            action_loss = action_criterion(act.transpose(1,2), labels[:,:,0].long())
            target_loss = target_criterion(tar.transpose(1,2), labels[:,:,1].long())
            loss = loss + action_loss + target_loss
        else:
            act,tar = model(inputs, input_lens)
            
            action_loss = action_criterion(act[:,:labels.size(1)].transpose(1,2), labels[:,:,0].long())
            target_loss = target_criterion(tar[:,:labels.size(1)].transpose(1,2), labels[:,:,1].long())
            loss = loss + action_loss + target_loss
        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        output = torch.cat((torch.argmax(act[:,:labels.size(1)],dim=2,keepdim=True),torch.argmax(tar[:,:labels.size(1)],dim=2,keepdim=True)),dim=2)
        em = (output == labels).long()
        #print(em)
        #print(output[1])
        #print(labels[1])
        #prefix = prefix_match(output, labels)
        acc = 0.0
        pm = 0.0
        for i in range(len(output)):
            acc += torch.mean(em[i,:labels_lens[i]],dtype=torch.float32)
            pm += prefix_match(output[i], labels[i])
        # logging
        epoch_loss += loss.item()
        epoch_acc += acc.item()/len(output)
        epoch_pm_acc += pm/len(output)

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)
    epoch_pm_acc /= len(loader)
    return epoch_loss, epoch_acc, epoch_pm_acc


def validate(args, model, loader, optimizer, action_criterion,target_criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc, val_pm_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )
    model.train()
    return val_loss, val_acc, val_pm_acc


def train(args, model, loaders, optimizer, action_criterion,target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    train_losses=[]
   
    train_accuracy=[]
    train_prefix_acc=[]

    val_losses = []
    
    val_accuracy = []
    val_prefix_acc = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc, train_pm_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss} | train acc: {train_acc} | train_pm_acc: {train_pm_acc}")

        train_losses.append(train_loss)
        #train_target_losses.append(train_target_loss)
        train_accuracy.append(train_acc)
        #train_target_accuracy.append(train_target_acc)
        train_prefix_acc.append(train_pm_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc, val_pm_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc} | val_pm_acc: {val_pm_acc}")
            val_losses.append(val_loss)
        #train_target_losses.append(train_target_loss)
            val_accuracy.append(val_acc)
        #train_target_accuracy.append(train_target_acc)
            val_prefix_acc.append(val_pm_acc)
        else:
            val_losses.append(val_losses[-1])
        #train_target_losses.append(train_target_loss)
            val_accuracy.append(val_accuracy[-1])
        #train_target_accuracy.append(train_target_acc)
            val_prefix_acc.append(val_prefix_acc[-1])

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #
    fig, axs = plt.subplots(2, 2,figsize=(10,8))
    axs[0, 0].plot(val_losses, 'b', label='Validation Loss')
    axs[0, 0].plot(train_losses,'g',label='Training Loss')
    axs[0, 0].set_title('loss')
    #axs.set(xlabel='Epochs', ylabel='Loss')
    #axs.set(xlabel='Epochs', ylabel='Loss')
    axs[0, 1].plot(val_accuracy, 'b', label='Validation exact_token accuracy')
    axs[0, 1].plot(train_accuracy,'g',label='Training exact_token accuracy')
    axs[0, 1].set_title('Exact_token Accuracy')
    #axs.set(xlabel='Epochs', ylabel='Accuracy')
    axs[1, 0].plot(val_prefix_acc, 'b', label='Validation prefix_match accuracy')
    axs[1, 0].plot(train_prefix_acc,'g',label='Training prefix_match accuracy')
    axs[1, 0].set_title('Prefix_match Accuracy')
    #axs.set(xlabel='Epochs', ylabel='Accuracy')
    fig.tight_layout()
    plt.show()

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps, max_episode_length = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, len(maps[0]),len(maps[1]),len(maps[3]))
    model = model.to(device)
    print(model)

    # get optimizer and loss functions
    action_criterion,target_criterion, optimizer = setup_optimizer(args, model)



    if args.eval:
        val_loss, val_acc, val_pm_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, action_criterion,target_criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument(
      "--emb_dim", default = 100, help=" specify the embedding dimension"
    )
    parser.add_argument(
      "--lstm_dim", default = 128, help=" sepcify lstm dimension (hidden size)"
    )
    parser.add_argument(
      "--num_layers", default = 1, help="specify number of lstm layers"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)