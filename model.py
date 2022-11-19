import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self,device,vocab_size,embedding_dim,hidden_dim,n_layers):
        super().__init__()
        self.device=device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #model_architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)


    def forward(self, x,x_length):
        #embedding_input = (batch_size, sequence_length)
        embeds = self.embedding(x)
        #embeds = embeds.transpose(0,1).continguous()
        #implement pack padded sequence before passing into lstm
        packed_embeddings = pack_padded_sequence(embeds, x_length, batch_first=True,enforce_sorted=False)
        
        lstm_out, (h0,c0) = self.lstm(packed_embeddings)
        return h0,c0


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self,device,output_dim1, output_dim2, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #model architecture
        self.embedding1 = nn.Embedding(output_dim1, embedding_dim, padding_idx=0)#embedding for actions
        self.embedding2 = nn.Embedding(output_dim2, embedding_dim, padding_idx=0)#embedding for targets
        self.lstm = nn.LSTM(embedding_dim*2, hidden_dim, n_layers, batch_first=True)
       
        
    def forward(self, input1, input2, input_length, hidden, cell):
        #input1, input2=input[:,:,0],input[:,:,1] #splitting input into action and targets
        embeds1 = self.embedding1(input1.long()) # B, N_y, e1
        embeds2 = self.embedding2(input2.long()) # B, N_y, e2
        
        concat_embeds = torch.cat((embeds1,embeds2),dim=2) # B, N, e1 + e2
        #print(concat_embeds.size())
        packed_concat_embeds = pack_padded_sequence(concat_embeds,input_length,batch_first=True, enforce_sorted=False) 
        lstm_out,(h0,c0) =  self.lstm(packed_concat_embeds, (hidden,cell))
        lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True) 
        return lstm_out, (h0,c0), lengths

class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, encoder, decoder, hidden_dim, output_dim1, output_dim2):
        super().__init__()
        
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.output1_dim = output_dim1
        self.output2_dim = output_dim2
        self.fc1 = nn.Linear(hidden_dim, output_dim1)
        self.fc2 = nn.Linear(hidden_dim, output_dim2)
        
        #get hidden state from lstm encoder
        #for loop for decoding iteratively
    def forward(self,x,x_length, y=None,y_length=None, max_seq_len = 25): 
        h_enc, c_enc = self.encoder(x,torch.tensor(x_length))
        #result = torch.zeros((y.size(0), y.size(1),self.output_dim)) #tensor to store decoder outputs
        if y is not None: #teacher forcing if gt are provided as input
            act = y[:,:,0]
            tar = y[:,:,1]
            action_result = torch.zeros((y.size(0), y.size(1), self.output1_dim),device=self.device) #tensor to store outputs
            target_result = torch.zeros((y.size(0), y.size(1), self.output2_dim),device=self.device)
            lstm_out,(h0,c0), dec_length = self.decoder(act,tar,y_length, h_enc, c_enc)
            for i in range(lstm_out.size(1)): #lstm_out.size(1) gives the max_sequence length for each bat
                #out1 = self.fc1(lstm_out[:,i,:])#action
                #out2 = self.fc2(lstm_out[:,i,:])#targets
                #merged_out = torch.cat((out1,out2), dim=2)#concatenating
                #result[:,i,:] = merged_out
                action_result[:,i,:] = self.fc1(lstm_out[:,i,:])
                target_result[:,i,:] = self.fc2(lstm_out[:,i,:])
            return action_result, target_result
        else: #student forcing where predicted sequence is fed into the next time step. Used during eval.
            a = torch.zeros((x.size(0),1),device= self.device)
            
            action_result = torch.zeros((x.size(0), max_seq_len, self.output1_dim),device=self.device) #tensor to store outputs
            target_result = torch.zeros((x.size(0), max_seq_len, self.output2_dim),device=self.device)
            temp_action = torch.zeros((x.size(0), 1),device = self.device)
            temp_target = torch.zeros((x.size(0), 1),device = self.device)
            temp_length_list = torch.ones((x.size(0))) #length of each prediction is only 1 i.e. one action and one target for each pair of 
            #action target pair.
            h_dec, c_dec = h_enc, c_enc # passing hidden state of encoder into decoder for first time step.
            for i in range(max_seq_len): #iterating over the max length in a batch
                dec_lstm_out, (h_dec, c_dec), lengths = self.decoder(temp_action, temp_target, temp_length_list, h_dec, c_dec) #decoder takes in the predicted action target values
                action_result[:,i,:] = self.fc1(dec_lstm_out[:,0,:]) #dec_lstm_out has shape(batch_size, 1, hidden_dim)
                target_result[:,i,:] = self.fc2(dec_lstm_out[:,0,:]) #dec_lstm_out[:,0,:] has shape(batch_size, hidden_dim) which is input shape for linear layer 
                temp_action = torch.argmax(action_result[:,i,:], dim=1, keepdim=True)
                temp_target = torch.argmax(target_result[:,i,:], dim=1, keepdim=True)
                a+=(temp_action==2)
                if(torch.all(a>0)): #To check if ending has been reached for each sequence in a batch
                    break
            return action_result, target_result





            


            

            


