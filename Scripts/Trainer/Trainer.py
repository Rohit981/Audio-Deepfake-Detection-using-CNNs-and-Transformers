import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data.dataloader as dataloader
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import Augmentation

class ModelTrainer:
    def __init__(self, model, device, loss_fn,learning_rate, 
                 batch_size,sampler, save_dir, model_name, start_from_checkpoint=False):
        super(ModelTrainer,self).__init__()
        
        #Intialize variables for training and evaluation
        self.optimizer = None
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.start_epoch = 0
        self.sampler = sampler
        self.best_valid_acc = 0
        self.lr_history = []

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        #Set Optimizer
        self.set_optimizer()

        #Initilaize a LR scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                            T_max=50,
                                            eta_min=1e-6)
        
        #Create save path
        self.save_path = os.path.join(save_dir, model_name + ".pt")
        self.save_dir = save_dir

        #Check if the dir exits, if not then make it
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        
        if start_from_checkpoint:
            self.Load_checkpoint()
        else:
            #If checkpoint doesn't exist and start_from_checkpoint is False
            #then raise an error
            if os.path.isfile(self.save_path):
                raise ValueError("Warning checkpoint exists")
            else:
                print("Starting from scratch")
        
        #Intialize Augmentation class for X data
        self.augmentation = Augmentation(frequency_mask_param=8,
                                         time_mask_param=12,
                                         noise_prob=0.4)


    #Set optimizer
    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-3)
    
    #Set data loader for test and train
    def set_data(self, train_set, test_set, val_set):
        print(f"Number of training examples: {len(train_set)}")
        print(f"Number of testing examples: {len(test_set)}")
        print(f"Number of Val examples: {len(val_set)}")


        self.train_loader = dataloader.DataLoader(train_set, batch_size=self.batch_size,sampler=self.sampler, num_workers=0, pin_memory=True)
        self.test_loader = dataloader.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.val_loader = dataloader.DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    #Load checkpoint and start from best epoch
    def Load_checkpoint(self):
        #Check if checkpoint exists
        if os.path.isfile(self.save_path):
            #Load checkpoint
            checkpoint = torch.load(self.save_path)

            #Checkpoint is stored as python dictionary
            #Here we unpack the dictionary to get our previous training states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            self.start_epoch = checkpoint['epoch']
            self.best_valid_acc = checkpoint['best_valid_acc']

            self.train_loss = checkpoint['train_loss']
            self.train_acc = checkpoint['train_acc']
            self.val_acc = checkpoint['val_acc']
            self.val_loss = checkpoint['val_loss']

            print("Checkpoint Loaded starting from epoch:", self.start_epoch)
        else:
            raise ValueError("Chekpoint Doesn't exist")
    
    #Save Checkpoint
    def save_checkpoint(self, epoch, valid_acc):
        self.best_valid_acc = valid_acc

        torch.save({
            "epoch" : epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            'lr_scheduler_state_dict':self.scheduler.state_dict(),
            "best_valid_acc": self.best_valid_acc,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_acc" : self.val_acc,
            "val_loss": self.val_loss
        }, self.save_path)


    #This function will perform single training epoch using our training data
    def Training(self):

        #Check for if training dataset is available
        if self.train_loader == None:
            print("Training Dataset loader not set")
        
        self.model.to(self.device)
        epoch_train_loss = 0
        #Train
        self.model.train()
        for i, (X,Y) in enumerate(tqdm(self.train_loader, leave=False, desc="Training")):
            #Set X and Y to device
            X,Y = X.to(self.device), Y.to(self.device)

            X = self.augmentation.forward(X)

            #Forward pass
            y_pred = self.model(X)

            #Set Loss
            loss = self.loss_fn(y_pred, Y.unsqueeze(1))
            
            #Zero out gradient
            self.optimizer.zero_grad()

            #Backpropogation and set gradient
            loss.backward()

            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            #Record the Learning Rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

            #Optimization
            self.optimizer.step()

            #Keep track of train loss for plotting
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(self.train_loader)

        self.train_loss.append(epoch_train_loss)
        self.scheduler.step() 
        
    #Evaluation of model this runs per one epoch
    def eval_model(self, train_test_val = "test"):
        #Check for data loader
        if self.test_loader is None:
            print(f"No test loader available")
        
        self.model.to(self.device)

        loader = None 
        state = "Evaluating"
        if train_test_val == "test":
            loader = self.test_loader
            state += "test"
        elif train_test_val == "train":
            loader = self.train_loader
            state += "train"
        elif train_test_val == "val":
            loader = self.val_loader
            state += "val"
        else:
            ValueError("Invalid Dataset, train_test should be train/test")
        
        #Evaluation and Initialize variables
        epoc_acc = 0
        epoch_loss = 0
        correct_pred = 0
        sample = 0
        all_preds = []
        all_labels = []
        all_probs = []
        f1,recall = 0,0

        self.model.eval()
        with torch.inference_mode():
            for i, (X,Y) in enumerate(tqdm(loader,leave=False, desc=state)):
                #Set X, Y to device
                X,Y = X.to(self.device), Y.to(self.device)

                #Forward pass
                fx = self.model(X)

                #Loss
                loss = self.loss_fn(fx, Y.unsqueeze(1))
                epoch_loss += loss.item()

                #Log sum of acc for BCE
                probs = torch.sigmoid(fx)
                preds = (probs > 0.5).float()
                # epoc_acc += (preds.squeeze(1) == Y).sum().item()
                correct_pred += (preds.squeeze(1) == Y).sum().item()
                sample += Y.size(0)

                #Calculate preds and labels for F1 and recall score
                all_preds.extend(preds.squeeze(1).cpu().numpy().astype(int))
                all_labels.extend(Y.cpu().numpy().astype(int))

                #Calculate probs for ROC Curve and auc score
                all_probs.extend(probs.squeeze(1).cpu().numpy())

        
        epoc_acc = correct_pred/sample
        epoch_loss /=len(loader)
        # self.all_preds = all_preds
        # self.all_labels = all_labels
        # self.all_probs = all_probs
        
        #Log accuracy, loss, F1 and recall from the epoch
        if train_test_val == "train":
            self.train_acc.append(epoc_acc)
        elif train_test_val == "val":
            self.val_acc.append(epoc_acc)
            self.val_loss.append(epoch_loss)
            f1 = f1_score(all_labels,all_preds, zero_division=0)
            recall = recall_score(all_labels,all_preds, zero_division=0)
        elif train_test_val == "test":
            f1 = f1_score(all_labels,all_preds, zero_division=0)
            recall = recall_score(all_labels,all_preds, zero_division=0)
        
        return epoch_loss,epoc_acc,f1,recall,all_preds,all_labels,all_probs
    
    def forward(self,x):
        return self.model(x)
        