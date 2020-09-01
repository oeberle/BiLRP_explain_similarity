import torch
import torch.nn as nn
import torch, torch.nn as nn, torch.nn.functional as F
import numpy
from torchvision import datasets, models, transforms
from utils import pool, newlayer, Flatten, Identity
import numpy as np
import os

def vgg_gamma(i):
    '''Setting gamma according to vgg layer index i''' 
    if i <=10:        gamma=0.5
    if 11 <= i <= 17: gamma=0.25
    if 18 <= i <= 24: gamma=0.1
    if i > 24:        gamma=0.0
    return gamma


class VggLayers(nn.Module):
    def __init__(self, feature_layer, h,w, embedding_size = 100, proj_case='random', device='cuda', seed=1):
        super(VggLayers, self).__init__()
        self.feature_layer = feature_layer
        self.h, self.w =  h,w
        self.proj_case = proj_case 
        self.embedding_size = embedding_size
        self.ls = [5,10,17,24,31]
        self.device = device
        self.filter_dict = {'5':64, '10':128, '17':256,  '24':512, '31':512}
        self.size_dict = {'5':2,'10':4, '17':8,  '24':16, '31':32}  
        
        torch.manual_seed(seed)
        
        if int(self.feature_layer) not in self.ls:
            raise ValueError('Please choose vgg-16 feature_layer from {}.'.format(list(map(str, self.ls))))
            
        if int(self.feature_layer) in self.ls:  
            self.h_proj, self.w_proj = int(self.h/self.size_dict[self.feature_layer]), int(self.w/self.size_dict[self.feature_layer])
            self.encoder= self.vgg_layer(int(self.feature_layer))   
            self.proj_dims = int(self.h_proj*self.w_proj*self.filter_dict[self.feature_layer])
            
        if self.proj_case == 'random':
            self.weight = torch.randn((self.proj_dims,self.embedding_size),  requires_grad=True).to(self.device)
            self.project = self.projection_conv(input_dim=self.proj_dims)
            self.project[0].weight = torch.nn.Parameter(self.weight,requires_grad=True )
                   
                    
    def identity(self):
        return torch.nn.Sequential([Identity()])
    
    def vgg_layer(self,layer):
        vgg_orig = models.vgg16(pretrained=True)
        layers = list(vgg_orig.features)[:layer] 
        return torch.nn.Sequential(*layers)      
    
    def projection_conv(self, input_dim):                         
        pca = torch.nn.Sequential(*[Flatten(), nn.Conv2d(input_dim, self.embedding_size , (1,1), bias=False),])
        return pca    
   
        
    def lrp(self, x, r, gamma_func):
        # Computig LRP relevances usig gamma-LRP
        
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1).to(self.device)
        std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1).to(self.device)

        lbound = (0-mean) / std
        hbound = (1-mean) / std
        
        if int(self.feature_layer) in self.ls:
            feature_layers, project_layers =  list(self.encoder) , list(self.project) 
            gamma_layers = [True]*len(feature_layers) + [False]*len(project_layers)
            layers = feature_layers + project_layers

        # Forward pass
        X   = [x.data]+[None for l in layers]
        for i,layer in enumerate(layers): 
            X[i+1] = layer.forward(X[i]).data

        # Backward pass
        for i,layer in list(enumerate(layers))[::-1]:
            
            x = X[i].clone().detach().requires_grad_(True)
           
            # Set gamma=0. for projection layers
            gamma = gamma_func(i)  if gamma_layers[i] else 0.

            # Handling projection layer
            if isinstance(layer,Flatten): 
                if self.proj_case == 'random':
                    if int(self.feature_layer) in self.ls:
                        r = r.view((1,self.filter_dict[self.feature_layer],self.h_proj,self.w_proj))
                        continue
                   
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.AvgPool2d) or isinstance(layer,nn.MaxPool2d):
                if i>0:
                    # Handling intermediate Conv2D or AvgPool layers 
                    z = newlayer(layer,lambda p: p + gamma*p.clamp(min=0)).forward(x)
                    z = z  + 1e-9
                    (z*(r/z).data).sum().backward()
                    r = (x*x.grad).data                    

                    
                else:
                    # Input Conv2D layer
                    l = (x.data*0+lbound).clone().detach().requires_grad_(True)
                    h = (x.data*0+hbound).clone().detach().requires_grad_(True)
                    z = layer.forward(x)-newlayer(layer,lambda p: p.clamp(min=0)).forward(l)-newlayer(layer,lambda p: p.clamp(max=0)).forward(h)
                    (z*(r/(z+1e-6)).data).sum().backward()
                    r = (x*x.grad+l*l.grad+h*h.grad).data
        return r
    
    def forward(self, x):
        h_0 = self.encoder(x)

        self.feature_shape = h_0.shape
        self.feature_data = h_0
        
        # Flatten vgg feature maps
        if self.proj_case == 'random':
            if int(self.feature_layer) in self.ls:
                h_0_flat = h_0.view((self.h_proj*self.w_proj*self.filter_dict[self.feature_layer],)).unsqueeze(0)
            h_0_flat = h_0.unsqueeze(2).unsqueeze(3)

        # Apply projection 
        o = self.project(h_0_flat)
        return h_0_flat, o
    
         
    def compute_branch(self, x, gamma_func=None):
        # Compute embeddings
        h, e = self.forward(x) 
        y = e.squeeze()
        n_features = y.shape
        
        # Computing LRP maps for each embedding dimension
        R = []
        for k, yk in  enumerate(y):
            z = np.zeros((n_features[0]))
            z[k] = y[k].detach().cpu().numpy().squeeze()
            r_proj =  torch.FloatTensor((z.reshape([1,n_features[0],1,1]))).to(self.device).data
            r = self.lrp(x, r_proj, gamma_func).detach().cpu().numpy().squeeze()
            R.append(r)
        return R  
    
    
    def bilrp(self, x1,x2, poolsize, gamma_func=None):
        # Computing LRP passes for each branch
        r1 = self.compute_branch(x1, gamma_func)
        r2 = self.compute_branch(x2, gamma_func)
        
        # Computing BiLRP relevances
        R = [np.array(r).sum(1) for r in [r1,r2]]
        R = np.tensordot(pool(R[0],poolsize),pool(R[1],poolsize), axes=(0,0))

        return R
    
  