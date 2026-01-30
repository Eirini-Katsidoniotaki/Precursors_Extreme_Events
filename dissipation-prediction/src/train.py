import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.losses import OutputWeightedMAE


class ModelTrain:
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5, alpha=1.0, extreme_threshold=0.2, pdf_func=None):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.extreme_threshold = extreme_threshold
        self.pdf_func = pdf_func
        
        # init optimizer, scheduler, and criterion
        self.optimizer = self._select_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor=0.5, patience=5, verbose=True)
        self.criterion = self._select_criterion()

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def _select_criterion(self):
        # Choose your loss here
        criterion = OutputWeightedMAE(self.pdf_func)
        return criterion

        
    
    def train_informer(self, train_loader, val_loader, pred_len, label_len, n_epochs, save_dir, model_name, use_amp = True):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_name)
        
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        self.model.to(self.device)
        train_losses, val_losses = [], []

        for epoch in range(n_epochs):
            # ---- Train ----
            self.model.train()
            total_train = 0.0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
        
                self.optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        #y_pred = self.model(xb)
                        #loss = self.criterion(y_pred, yb.squeeze(-1))
                        pred, true = self._process_one_batch(xb, yb, pred_len, label_len)
                        loss = self.criterion(pred, true)
        
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    #y_pred = self.model(xb)
                    #loss = self.criterion(y_pred, yb.squeeze(-1))
                    pred, true = self._process_one_batch(xb, yb, pred_len, label_len)
                    loss = self.criterion(pred, true)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
        
                total_train += loss.item() * xb.size(0)

            avg_train = total_train / len(train_loader.dataset)
            train_losses.append(avg_train)

            # ---- Validate ----
            self.model.eval()
            total_val = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    #y_pred = self.model(xb)
                    #val_loss = self.criterion(y_pred, yb.squeeze(-1))
                    pred, true = self._process_one_batch(xb, yb, pred_len, label_len)
                    val_loss = self.criterion(pred, true)
                    
                    total_val += val_loss.item() * xb.size(0)

            avg_val = total_val / len(val_loader.dataset)
            val_losses.append(avg_val)
            self.scheduler.step(avg_val)

            print(f"Epoch {epoch+1:02d} | train {avg_train:.5f} | val {avg_val:.5f}")

        # Save model
        torch.save(self.model.state_dict(), save_path)
        print(f"\n [INFO] Best model saved to {save_path}")

        # Save history
        np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_dir, "val_losses.npy"),   np.array(val_losses))

        # Plot
        plt.figure(figsize=(8,3))
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
        plt.close()
        
        
    def _process_one_batch(self, batch_x, batch_y, pred_len = 200, label_len = 200, padding = 0, output_attention = False, use_amp = False, features = 'S', inverse = False):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
    
        
            # decoder input
            if padding==0:
                #dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]], dtype=torch.float32, device=self.device)
            elif padding==1:
                #dec_inp = torch.ones([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
                dec_inp = torch.ones([batch_y.shape[0], pred_len, batch_y.shape[-1]], dtype=torch.float32, device=self.device)

            dec_inp = torch.cat([batch_y[:,:label_len,:], dec_inp], dim=1) #.float().to(self.device)
            
            
            # encoder - decoder
            if use_amp:
                #with torch.cuda.amp.autocast():
                with torch.amp.autocast('cuda'):
                    if output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)
            else:
                if output_attention:
                    outputs = self.model(batch_x, dec_inp)[0]
                else:
                    outputs = self.model(batch_x, dec_inp)
            
            #if inverse:
            #    outputs = dataset_object.inverse_transform(outputs)
            f_dim = -1 if features=='MS' else 0
            batch_y = batch_y[:,-pred_len:,f_dim:].to(self.device)
        
            return outputs, batch_y  
