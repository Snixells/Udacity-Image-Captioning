import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
       
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # LSTM Cell first => Also try Embedding first, Maybe also Dropout
        self.lstm = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        
        # Fully Connected Layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # Embedding Layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # Activation
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
        
        captions_embed = self.embed(captions)
        
        for t in range(captions.size(1)):
            if t == 0:
                hidden_state, cell_state = self.lstm(features, (hidden_state, cell_state))
            else:
                hidden_state, cell_state = self.lstm(captions_embed[:, t, :], (hidden_state, cell_state))
               
            out = self.fc(hidden_state)
            
            outputs[:, t, :] = out
            
        return outputs
    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predictions = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            out = lstm_out.squeeze(1)
            out = self.fc(out)
            _, prediction = out.max(1)
            outputs.append(prediction.item())
            
            if prediction == 1:
                break
            
            embeddings = self.embed(prediction).unsqueeze(1)
      
        return outputs
        # pass