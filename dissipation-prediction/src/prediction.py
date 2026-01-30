import os
import numpy as np
import torch
import matplotlib.pyplot as plt


class Predict:
    def __init__(self, model, device, test_loader,
                 save_dir="./checkpoints/LSTM",
                 model_name="LSTMForecaster.pth",
                 results_path="./results/LSTM"):

        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.model_name = model_name
        self.results_path = results_path
        self.preds = None
        self.trues = None

        # Load model
        best_path = os.path.join(save_dir, model_name)
        self.model.load_state_dict(torch.load(best_path, map_location=device))
        self.model.eval()
        print(f"[INFO] Loaded model from {best_path}")


    def run_informer(self, pred_len, label_len):
        #all_preds, all_trues = [], []
        preds = []
        trues = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb = xb.to(self.device)     # (B, m, F)
                #y_pred = self.model(xb)     # (B, out_len)
                pred, true = self._process_one_batch(xb, yb, pred_len, label_len)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                #all_preds.append(y_pred.cpu())
                #all_trues.append(yb.cpu())

        # Stack all batches
        #self.preds = torch.cat(all_preds, dim=0).numpy()  # (Ntest, out_len)
        #self.trues = torch.cat(all_trues, dim=0).numpy()  # (Ntest, out_len, 1) if squeeze not applied
        
        # Stack all batches along batch axis
        self.preds = np.concatenate(preds, axis=0)   # (Ntest, pred_len, 1)
        self.trues = np.concatenate(trues, axis=0)   # (Ntest, pred_len, 1)
        print('Test shape:', self.preds.shape, self.trues.shape)
        
        #self.preds = np.array(preds)
        #self.trues = np.array(trues)
        #print(Ttest shape:', self.preds.shape, self.trues.shape)
        #self.preds = self.preds.reshape(-1, self.preds.shape[-2], self.preds.shape[-1])
        #self.trues = self.trues.reshape(-1, self.trues.shape[-2], self.trues.shape[-1])
        #print('Test shape:', self.preds.shape, self.trues.shape)


        #print("Test shapes:", self.preds.shape, self.trues.shape)

        # Save results
        os.makedirs(self.results_path, exist_ok=True)
        np.save(os.path.join(self.results_path, "pred.npy"), self.preds)
        np.save(os.path.join(self.results_path, "true.npy"), self.trues)
        print(f"[INFO] Predictions saved to {self.results_path}")

        return self.preds, self.trues
        
        
        
    def plot(self, k_list, DT):
        # Ensure consistent shape
        preds = self.preds
        trues = self.trues
        if preds.ndim == 2: preds = preds[..., None]
        if trues.ndim == 2: trues = trues[..., None]
        
        for k in k_list:
            pred_h = preds[:, k, 0]         
            true_h = trues[:, k, 0]
            
            t_ahead = (k+1) * DT
            time = np.arange(len(pred_h)) * DT
            
            
            plt.figure(figsize=(10,4))
            plt.plot(time, true_h, label="True")   # [0:10000]
            plt.plot(time, pred_h, label="Predicted")
            plt.title(f"Horizon t+{t_ahead:.2f} s")
            plt.legend()
            
            # Save in the same folder
            save_path = os.path.join(self.results_path, f"horizon_t_{t_ahead:.2f}_plot.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Plot saved to {save_path}")
            
            
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
        
