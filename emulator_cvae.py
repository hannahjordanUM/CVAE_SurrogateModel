import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class EncoderCVAE(nn.Module):
    def __init__(self,N,num_params,latent,hidden_dims):
        super(EncoderCVAE, self).__init__()
        
        self.l1 = nn.Linear(N+num_params,hidden_dims[0]) # map from data to first hidden layer 
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1]) # map from first hidden layer to second hidden layer
        
        self.l3_mu = nn.Linear(hidden_dims[1],latent) # map from second hidden layer to latent space to represent mean
        self.l3_sigma = nn.Linear(hidden_dims[1],latent) # map from second hidden layer to latent space to represent standard deviation
        
    def forward(self,u):
        
        a1 = self.l1(u)
        z1 = torch.relu(a1)
        
        a2 = self.l2(z1)
        z2 = torch.relu(a2)
        
        mu = self.l3_mu(z2)
        sigma = self.l3_sigma(z2)
        
        epsilon = torch.randn_like(sigma)
        z = mu + torch.exp(sigma/2.)*epsilon
        
        return z, mu, sigma
    
    
class DecoderCVAE(nn.Module):
    def __init__(self,N,num_params,latent,hidden_dims):
        super(DecoderCVAE, self).__init__()
        
        self.l1 = nn.Linear(latent+num_params,hidden_dims[1]) # map from data to first hidden layer 
        self.l2 = nn.Linear(hidden_dims[1], hidden_dims[0]) # map from first hidden layer to second hidden layer
        self.l3 = nn.Linear(hidden_dims[0],N) # map from second hidden layer to latent space to represent mean
        
    def forward(self,z):
        
        a1 = self.l1(z)
        z1 = torch.relu(a1)
        
        a2 = self.l2(z1)
        z2 = torch.relu(a2)
        
        u_prime = self.l3(z2)
        #u_prime = torch.relu(a3)
                
        return u_prime

    
    
class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self,N,num_params,latent,hidden_dims):
        super(ConditionalVariationalAutoencoder, self).__init__()
        self.encoder = EncoderCVAE(N,num_params,latent,hidden_dims)
        self.decoder = DecoderCVAE(N,num_params,latent,hidden_dims)
        
    def forward(self,u,params,train=True):  
        encoder_input = torch.cat((u,params),1)
        z, mu, sigma = self.encoder(encoder_input)
        decoder_input = torch.cat((z,params),1)
        u_prime = self.decoder(decoder_input)
        
        if train:
            return u_prime, mu, sigma
        else:
            return u_prime
        