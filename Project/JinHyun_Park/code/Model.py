### YOUR CODE HERE
# import tensorflow as tf
# import torch
import torch 
import torch.nn as nn 
from tqdm import tqdm
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record, preprocess_for_testing

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        self.network = MyNetwork(configs)
        self.network = self.network.cuda()
        self.learning_rate = 1e-2
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train(self, x_train, y_train, max_epochs):
        print("--- Training started ---")
        self.network.train()

        samples, _ = x_train.shape
        batches = samples // self.configs.batch_size 
        loss = 0

        for epoch in range(1, max_epochs + 1):
            time1 = time.time()

            shuffled_idx = np.random.permutation(samples)
            shuffled_x_train = x_train[shuffled_idx]
            shuffled_y_train = y_train[shuffled_idx]

            for b in range(batches):
                shuffled_x_batch = [parse_record(x, True) for x in shuffled_x_train[b * self.configs.batch_size: (b+1) * self.configs.batch_size]]
                shuffled_y_batch = shuffled_y_train[b * self.configs.batch_size: (b+1) * self.configs.batch_size]
                shuffled_x_batch_t = torch.stack((shuffled_x_batch)).float().cuda()
                shuffled_y_batch_t = torch.tensor((shuffled_y_batch)).float().cuda()
                
                model = self.network.cuda()
                y_preds = model(shuffled_x_batch_t)

                self.optimizer.zero_grad()
                loss = self.cross_entropy_loss(y_preds, shuffled_y_batch_t.long())
                loss.backward()
                self.optimizer.step()

                print(f"Batch: {b}/{batches} --- Loss: {loss}", end='\r', flush=True)

            time2 = time.time()  
            duration = time2 - time1
            print(f"Epoch: {epoch} --- Loss: {loss} --- Duration: {duration}")
            
            # write a file
            # contains epoch and loss
            with open('loss.txt', 'a') as writefile:
                writefile.write(f'{epoch}, {loss}\n')

            self.learning_rate_scheduler.step()

            if epoch % self.configs.save_interval == 0:
                self.save_model(epoch)


    def evaluate(self, x, y, checkpoints):
        print("--- Evaluation ---")
        self.network.eval()
        for checkpoint in checkpoints:
            checkpoint_model = os.path.join("../model_dir", 'my-model-%d.ckpt' % checkpoint)
            self.load_model(checkpoint_model)

            predictions = list()
            samples, _ = x.shape
            for i in tqdm(range(samples)):
                img = parse_record(x[i], False).float().to('cuda').view(1, 3, 32, 32)
                logits = self.network(img)
                pred = int(torch.max(logits.data, 1)[1])
                predictions.append(pred)

            print(f"Test accuracy: {torch.sum( torch.tensor(predictions) == torch.tensor(y) ) / torch.tensor(y).shape[0]}")


    # predict the probability on private data
    def predict_prob(self, x):
        print("--- Evaluation for Private Dataset ---")
        self.network.eval()
        self.load_model('../private_dir/mymodel.ckpt')

        predictions = list()
        samples, _ = x.shape
        for i in tqdm(range(samples)):
            img = x[i].reshape((32, 32, 3))
            # img = preprocess_for_testing(img).float().to('cuda').view(1, 3, 32, 32)
            img = preprocess_for_testing(img).float().to('cuda').view(1, 3, 32, 32)
            pred = self.network(img)
            predictions.append(pred.cpu().detach().numpy())
        
        predictions = np.array(predictions)
        x, y, z = predictions.shape
        print("x, y, z", x, y, z)
        predictions = predictions.reshape((x, y * z))
        exp_predictions = np.exp(predictions)
        summed_exp_predictions = exp_predictions.sum(axis=1)
        logits = (exp_predictions.T / summed_exp_predictions).T
        
        print("Verification: shape of predictions is", logits.shape)
        
        return np.array(logits)

    # the following is from HW2
    def save_model(self, epoch):
        pwd = os.path.join("../model_dir", 'my-model-%d.ckpt' % epoch)
        os.makedirs("../model_dir", exist_ok=True)
        torch.save(self.network.state_dict(), pwd)
        print(f"--- Saving model at epoch {epoch} ---")
    
    # the following is from HW2
    def load_model(self, checkpoint_model_epoch):
        loaded_model = torch.load(checkpoint_model_epoch, map_location='cpu')
        self.network.load_state_dict(loaded_model)
        print(f"--- Loading model {checkpoint_model_epoch} ---")
### END CODE HERE