from ..Feat_extraction import Feature_extraction_abs
from . import Discriminator
import torch
import numpy as np
import torch.nn.functional as F
class OSR_classifier:
    def __init__(self,classifier:Feature_extraction_abs,discriminator:Discriminator,epsilon):
        self.classifier = classifier
        self.discriminator = discriminator
        self.epsilon = epsilon

        

    def discriminate(self,x):   
        with torch.no_grad():
            open_set_likelihood = self.discriminator(x)

        return open_set_likelihood
        
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon
        
    def classify(self,X):
        #features = self.classifier.extract_features(X)
        features = X
        
        open_set_likelihood = self.discriminate(features)
        #print(open_set_likelihood[:5])
        classification_mask = open_set_likelihood >= self.epsilon
        
        with torch.no_grad():
            features = features.view(features.size(0), -1)
            
            logits = self.classifier.classify_features(features)
            softmax = F.softmax(logits,dim=1)
            predictions = torch.argmax(softmax, dim=1)
            
            
        #print(classification_mask.view(-1))
        final_predictions = torch.where(
            classification_mask.view(-1), 
            predictions, 
            -1
        )
        
        return final_predictions,open_set_likelihood.view(-1)

def main():
    open_set_likelihood = torch.tensor([0, 1, 0.5, 0.6, 0.2])
    epsilon = 0.5
    classification_mask = open_set_likelihood >= epsilon
    print(classification_mask)

if __name__  == "__main__":
    main()