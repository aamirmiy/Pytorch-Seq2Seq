# Pytorch - Sequence to Sequence 

You will implement an Encoder-Decoder model that takes in [ALFRED](https://askforalfred.com/) instructions for an entire episode and predicts the sequence of corresponding, high-level actions and target objects. The problem is the same as in Homework #1 except you will model this as a sequence prediction problem instead of multiclass classification. You will implement a sequence-to-sequence (seq2seq) model. 

ALFRED instructions were written by Mechanical Turk workers who aligned them to a virtual agent's behavior in a recorded simulation-based video of the agent doing a task in a room.

For example, the instructions for [an example task](https://askforalfred.com/?vid=8781) and their associated high-level action and object targets are:

| Instruction                                                                                                                          | Action       | Target     |
| :----------------------------------------------------------------------------------------------------------------------------------- | ------------:| ----------:|
| Go straight and to the left to the kitchen island.                                                                                   | GotoLocation | countertop |
| Take the mug from the kitchen island.                                                                                                | PickupObject | mug        |
| Turn right, go forward a little, turn right to face the fridge.                                                                      | GotoLocation | fridge     |
| Put the mug on the lowest level of the top compartment of the fridge. Close then open the fridge door. Take the mug from the fridge. | CoolObject   | mug        |
| Turn around, go straight all the way to the counter to the left of the sink, turn right to face the sink.                            | GotoLocation | sinkbasin  |
| Put the mug in the sink.                                                                                                             | PutObject    | sinkbasin  |

Initially, you should implement a encoder-decoder seq2seq model that encodes the low-level instructions into a context vector which is decoded autoregressively into the high-level instruction. Then you will implement an attention mechanism that allows the decoder to attend to each hidden state of the encoder model when making predictions. Finally, you will compare this against a Transformer-based model. You may use any functionality in the HuggingFace library for these implementations. (That is, we do not expect you to implement RNNs, LSTMs, Attention layers, Tranformers, or any other architectural component from scratch.)

We provide starter code that tokenizes the instructions and provides dictionaries mapping instruction tokens to their numerical indexes. It's up to you to write methods that convert the inputs and outputs to tensors, an encoder-decoder attention model that processes input tensors to produce predictions, and the training loop to adjust the parameters of the model based on its predictions versus the ground truth, target outputs. Note, you will need to implement some function for decoding the target text given the context vector. For this decoding to work, you will need to append special tokens to the input to mark the beginning of sentence (\<BOS\>) and end of sentence (\<EOS\>). This is a standard practice for decoder models. 

You will evaluate your model as it trains against both the training data that it is seeing and validation data that is "held out" of the training loop. 


## Install some packages

```
# first create a virtualenv 
virtualenv -p $(which python3) ./hw3

# activate virtualenv
source ./hw1/bin/activate

# install packages
pip3 install -r requirements.txt
```

## Train model

The training file will throw some errors out of the box. You will need to fill in the TODOs before anything starts to train.
While debugging, consider taking a small subset of the data and inserting break statements in the code and print the values of your variables.

```
Train:
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/s2s \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu 

Evaluation:
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/s2s \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu \
    --eval


# add any additional argments you may need
# remove force_cpu if you want to run on gpu
```


