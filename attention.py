import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
class AttEncoder(nn.Module):
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
        
        packed_embeddings = pack_padded_sequence(embeds, x_length, batch_first=True,enforce_sorted=False)
        
        lstm_out, (h0,c0) = self.lstm(packed_embeddings)
        return lstm_out,h0,c0


class AttDecoder(nn.Module): #with attention
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

        self.lstm = nn.LSTM(embedding_dim*2+hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.energy = nn.Linear(hidden_dim*2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
       
        
    def forward(self, input1, input2, input_length, lstm_out, hidden, cell):
        #input1, input2=input[:,:,0],input[:,:,1] #splitting input into action and targets
        embeds1 = self.embedding1(input1.long()) # B, N_y, e1
        embeds2 = self.embedding2(input2.long()) # B, N_y, e2

        padded_lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True)
        sequence_length = padded_lstm_out.size(1)
        hidden_permuted = hidden.permute(1,0,2)
        hidden_reshaped = hidden_permuted.repeat(1,sequence_length,1)
        
        energy = self.relu(self.energy(torch.cat((hidden_reshaped, padded_lstm_out),dim=2)))
        #(N, sequence_len, 1)
        attention = self.softmax(energy)
        #(N, sequence_len, 1)
        attention = attention.permute(0,2,1)
        #(N,1,sequence_length)
        #lstm_out = (N,sequence_len, hidden_size)
        context_vector = torch.bmm(attention, padded_lstm_out)
        #(N,1,hidden_size)
        concat_embeds = torch.cat((embeds1,embeds2),dim=2)
        #(N, 1, e1 + e2)
        
        rnn_input = torch.cat((context_vector, concat_embeds), dim=2)
        packed_rnn_input = pack_padded_sequence(rnn_input,input_length,batch_first=True, enforce_sorted=False)
        
        #(N,1, hidden+embed*2) 
        dec_lstm_out,(h0,c0) =  self.lstm(packed_rnn_input, (hidden,cell))
        dec_lstm_out, lengths = pad_packed_sequence(dec_lstm_out, batch_first=True) 
        return dec_lstm_out, (h0,c0), lengths

class AttEncoderDecoder(nn.Module):
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
    def forward(self,x,x_length, y=None,y_length=None, max_seq_len = 30): 
        enc_lstm_out, h_enc, c_enc = self.encoder(x,torch.tensor(x_length))
        
        if y is not None: #teacher forcing if gt are provided as input
            act = y[:,:,0]
            tar = y[:,:,1]
            action_result = torch.zeros((y.size(0), y.size(1), self.output1_dim),device=self.device) #tensor to store outputs
            target_result = torch.zeros((y.size(0), y.size(1), self.output2_dim),device=self.device)
            #lstm_out,(h0,c0), dec_length = self.decoder(act,tar,y_length, enc_lstm_out, h_enc, c_enc)
            temp_length_list = torch.ones((x.size(0)))
            h_dec, c_dec = h_enc, c_enc
            for i in range(y.size(1)): #y.size(1) gives the max_sequence length for each bat
              
                dec_lstm_out, (h_dec, c_dec), lengths = self.decoder(act[:,i].unsqueeze(1), tar[:,i].unsqueeze(1), temp_length_list, enc_lstm_out, h_dec, c_dec)
                action_result[:,i,:] = self.fc1(dec_lstm_out[:,0,:])
                target_result[:,i,:] = self.fc2(dec_lstm_out[:,0,:])
            return action_result, target_result
        else: #student forcing where predicted sequence is fed into the next time step. Used during eval.
            a = torch.zeros((x.size(0),1),device= self.device)
            
            action_result = torch.zeros((x.size(0), max_seq_len, self.output1_dim),device=self.device) #tensor to store outputs
            target_result = torch.zeros((x.size(0), max_seq_len, self.output2_dim),device=self.device)
            temp_action = torch.zeros((x.size(0), 1),device = self.device)
            temp_target = torch.zeros((x.size(0), 1),device = self.device)
            temp_length_list = torch.ones((x.size(0))) #length of each prediction is only 1 i.e. one action and one target for each pair of 
            #action target pair.
            dec_lstm_out,h_dec, c_dec = enc_lstm_out,h_enc, c_enc # passing hidden state of encoder into decoder for first time step.
            for i in range(max_seq_len): #iterating over the max length in a batch
                dec_lstm_out, (h_dec, c_dec), lengths = self.decoder(temp_action, temp_target, temp_length_list, enc_lstm_out, h_dec, c_dec) #decoder takes in the predicted action target values
                action_result[:,i,:] = self.fc1(dec_lstm_out[:,0,:]) #dec_lstm_out has shape(batch_size, 1, hidden_dim)
                target_result[:,i,:] = self.fc2(dec_lstm_out[:,0,:]) #dec_lstm_out[:,0,:] has shape(batch_size, hidden_dim) which is input shape for linear layer 
                temp_action = torch.argmax(action_result[:,i,:], dim=1, keepdim=True)
                temp_target = torch.argmax(target_result[:,i,:], dim=1, keepdim=True)
                a+=(temp_action==2)
                if(torch.all(a>0)):
                    break
            return action_result, target_result