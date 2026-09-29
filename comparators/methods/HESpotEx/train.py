import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.utils.data.distributed
from model.utils import get_free_gpu

from model.models import Image_Model,Image_Model_ST1K

from torch.utils.data import DataLoader
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    
def train_epoch(model, train_loader, optimizer, device):
    # loss_meter = AvgMeter()
    latent_simloss_mean=[]
    # PCC_loss_mean=[]
    mse_loss_mean=[]
    counts_loss_mean=[]
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batchs in tqdm_object:
        batch = batchs['meta']
        batch = {k: v.to(device) for k, v in batch.items() if k == "image" or 
                k == "expression" or k == "latent" or k == "adj" or k == "nuclei_number"}
        #batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "expression" or k == "latent" or k == "adj"}
        
        latent_simloss, mse_loss, _, _ = model(batch)
        loss = 1*latent_simloss + 1*mse_loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # lr_scheduler.step()

        latent_simloss_mean.append(latent_simloss.item())
        # PCC_loss_mean.append(PCC_loss.item())
        mse_loss_mean.append(mse_loss.item())
        # counts_loss_mean.append(counts_loss.item())

        # count = batch["image"].size(1)
        # loss_meter.update(loss.item(), count)

        # tqdm_object.set_postfix(train_loss=loss_meter.avg)
    latent_simloss_mean = sum(latent_simloss_mean) / len(latent_simloss_mean)
    # PCC_loss_mean = sum(PCC_loss_mean) / len(PCC_loss_mean)
    mse_loss_mean = sum(mse_loss_mean) / len(mse_loss_mean)
    # counts_loss_mean = sum(counts_loss_mean) / len(counts_loss_mean)
    return latent_simloss_mean, mse_loss_mean

def test_epoch(model, test_loader, device):
    latent_simloss_mean=[]
    PCC_loss_mean=[]
    mse_loss_mean=[]
    counts_loss_mean=[]
    predict=[]
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batchs in tqdm_object:       
        batch = batchs['meta']
        batch = {k: v.to(device) for k, v in batch.items() if k == "image" or 
                    k == "expression" or k == "latent" or k == "adj" or k == "nuclei_number"}
        #batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "expression" or k == "latent" or k == "adj"}
        latent_simloss, mse_loss, PCC_loss, out = model(batch)
        latent_simloss_mean.append(latent_simloss.item())
        PCC_loss_mean.append(PCC_loss.item())
        mse_loss_mean.append(mse_loss.item())
        # counts_loss_mean.append(counts_loss.item())

        predict.append(out.cpu().detach().numpy())
    
    latent_simloss_mean = sum(latent_simloss_mean) / len(latent_simloss_mean)
    PCC_loss_mean = sum(PCC_loss_mean) / len(PCC_loss_mean)
    mse_loss_mean = sum(mse_loss_mean) / len(mse_loss_mean)
    # counts_loss_mean = sum(counts_loss_mean) / len(counts_loss_mean)

    return latent_simloss_mean, PCC_loss_mean, mse_loss_mean, predict


def main(train_dataset, val_dataset, test_dataset, image_embedding, 
         out_embedding, max_epochs=50, save_path=None,method='global',
         device='cuda:1', return_metadata=False,
         checkpoint_selection='outer_test'):
    if checkpoint_selection not in {'outer_test', 'validation', 'fixed_final_epoch'}:
        raise ValueError(f"unknown checkpoint_selection: {checkpoint_selection}")
    if checkpoint_selection == 'validation' and val_dataset is None:
        raise ValueError("validation checkpoint selection requires val_dataset")
    seed=0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # try:
    #     gpu_index = get_free_gpu()
    #     print(f"Choose GPU:{gpu_index} as device")
    # except:
    #     print('GPU is not available')

    # device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    device=str(device)
    print("device: {}".format(device))

    if method=='ST1K':
        model = Image_Model_ST1K(latent_dim=image_embedding, out_embedding=out_embedding,method=method).to(device)
    else:
        model = Image_Model(latent_dim=image_embedding, out_embedding=out_embedding,method=method).to(device)
    
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # model = nn.DataParallel(model, device_ids=[2,3],output_device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )

    num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(max_epochs):
        print(f"Epoch: {epoch + 1}")
        # step = "epoch"

        # Train the model
        model.train()

        train_latent_simloss, train_mse_loss = train_epoch(model, train_loader, optimizer, device)


        # Evaluate only the data permitted to select this checkpoint.  In the
        # nested-refit mode the outer test is opened exactly once after the
        # final training epoch.
        model.eval()
        with torch.no_grad():
            if checkpoint_selection == 'validation':
                val_latent_simloss, val_PCC_loss, val_mse_loss, val_predict = test_epoch(
                    model, val_loader, device
                )
                selection_loss = val_PCC_loss
            elif checkpoint_selection == 'outer_test':
                test_latent_simloss, test_PCC_loss, test_mse_loss, predict = test_epoch(
                    model, test_loader, device
                )
                selection_loss = test_PCC_loss
            else:
                selection_loss = float('inf')
        
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, train_latent: {train_latent_simloss:.3f}, train_mse: {train_mse_loss:.3f}')
            if checkpoint_selection == 'validation':
                print(f'Epoch: {epoch:03d}, val_latent: {val_latent_simloss:.3f}, val_PCC: {val_PCC_loss:.3f}, val_mse: {val_mse_loss:.3f}')
            if checkpoint_selection == 'outer_test':
                print(f'Epoch: {epoch:03d}, test_latent: {test_latent_simloss:.3f}, test_PCC: {test_PCC_loss:.3f}, test_mse: {test_mse_loss:.3f}')
        if checkpoint_selection == 'fixed_final_epoch' or selection_loss < best_loss:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            best_loss = selection_loss
            # best_gene_PCC_loss = test_gene_PCC_loss
            best_epoch = epoch
            if checkpoint_selection == 'outer_test':
                best_predict = predict
            if checkpoint_selection == 'validation':
                best_val_predict = val_predict

            torch.save(model.state_dict(), save_path + "/best_unLSTM.pt")
            print(f"Saved Best Model! best_PCC_loss: {best_loss:.4f}")

    if checkpoint_selection in {'validation', 'fixed_final_epoch'}:
        model.load_state_dict(torch.load(save_path + "/best_unLSTM.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            test_latent_simloss, test_PCC_loss, test_mse_loss, best_predict = test_epoch(
                model, test_loader, device
            )
        if checkpoint_selection == 'fixed_final_epoch':
            best_loss = test_PCC_loss

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    for i in range(len(best_predict)):
        best_predict[i] = best_predict[i]-best_predict[i].min()
    if checkpoint_selection == 'validation':
        for i in range(len(best_val_predict)):
            best_val_predict[i] = best_val_predict[i]-best_val_predict[i].min()
        
        if return_metadata:
            return (best_predict, best_val_predict), {
                "best_epoch_zero_based": int(best_epoch),
                "best_epoch_one_based": int(best_epoch + 1),
                "best_validation_pcc_loss": float(best_loss),
                "checkpoint_selected_on_outer_test": False,
                "checkpoint_selection": "validation",
            }
        return best_predict, best_val_predict
    if return_metadata:
        return best_predict, {
            "best_epoch_zero_based": int(best_epoch),
            "best_epoch_one_based": int(best_epoch + 1),
            "best_test_pcc_loss": float(best_loss),
            "checkpoint_selected_on_outer_test": checkpoint_selection == 'outer_test',
            "checkpoint_selection": checkpoint_selection,
        }
    return best_predict
