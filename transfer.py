import os
import numpy as np
import torch
import argparse
from model import SASNet
import warnings
import random
from datasets.loading_data import loading_data
import matplotlib.pyplot as plt
import csv

warnings.filterwarnings('ignore')

def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--model_path', type=str, help='path of pre-trained model')
    parser.add_argument('--data_path', type=str, help='root path of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--log_para', type=int, default=1000, help='magnify the target density map')
    parser.add_argument('--block_size', type=int, default=32, help='patch size for feature level selection')

    return parser

# get the dataset
def prepare_dataset(args):
    return loading_data(args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # update the moving average
    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count
    
def main(args):
    """the main process of inference"""

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_loader, val_loader = prepare_dataset(args)
    
    model = SASNet(args=args).cuda()
    # load the trained model
    model.load_state_dict(torch.load(args.model_path))
    print('successfully load model from', args.model_path)
    
    #set the training parameter
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    best_val_mae = float('inf')
    
    mae_epoch = []
    mse_epoch = []
    mae_batch = []
    mse_batch = []
    val_mae_epoch = []
    val_mse_epoch = []
    val_mae_batch = []
    val_mse_batch = []
    best_val_mae = float('inf')    
    for epoch in range(50):
        maes = AverageMeter()
        mses = AverageMeter()
        # iterate over the dataset
        for vi, data in enumerate(train_loader):
            img, gt_map = data
            if torch.cuda.is_available():
                img = img.cuda()
                gt_map = gt_map.type(torch.FloatTensor).cuda()

            # train the model
            optimizer.zero_grad()
            pred_map = model(img)

            loss = criterion(pred_map/1000, gt_map)
            loss.backward()
            optimizer.step()
            
            model.eval()
            
            pred_map = pred_map.cpu().detach().numpy()
            gt_map = gt_map.cpu().detach().numpy()
            # evaluation over the batch
            for i_img in range(pred_map.shape[0]):
                #print(gt_map.shape)
                pred_cnt = np.sum(pred_map[i_img], (1, 2)) / args.log_para
                gt_count = np.sum(gt_map[i_img], (1, 2))
                mae = abs(gt_count - pred_cnt)
                mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)
                mae_batch.append(mae)
                mse_batch.append(mse)
                maes.update(mae)
                mses.update(mse)
            model.train()
        scheduler.step()
        train_mae = maes.avg
        train_mse = np.sqrt(mses.avg)
        mae_epoch.append(train_mae)
        mse_epoch.append(train_mse)
        
        #validation loop        
        val_maes = AverageMeter()
        val_mses = AverageMeter()

        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(val_loader):
                v_img, v_gt_map = v_data
                if torch.cuda.is_available():
                    v_img = v_img.cuda()
                    #gt_map = gt_map.type(torch.FloatTensor).cuda()
                    
                v_pred_map = model(v_img)        
                v_pred_map = v_pred_map.cpu().detach().numpy()              
                #v_gt_map = gt_map.cpu().detach().numpy()
                v_gt_map = v_gt_map.numpy()
                # evaluation over the batch
                for vi_img in range(v_pred_map.shape[0]):
                    v_pred_cnt = np.sum(v_pred_map[vi_img], (1, 2)) / args.log_para
                    v_gt_count = np.sum(v_gt_map[vi_img], (1, 2))
                    v_mae = abs(v_gt_count - v_pred_cnt)
                    v_mse = (v_gt_count - v_pred_cnt) * (v_gt_count - v_pred_cnt)
                    val_mae_batch.append(v_mae)
                    val_mse_batch.append(v_mse)
                    val_maes.update(v_mae)
                    val_mses.update(v_mse)
        # calculation mean validation mae and mse
        val_mae = val_maes.avg
        val_mse = np.sqrt(val_mses.avg)
        val_mae_epoch.append(val_mae)
        val_mse_epoch.append(val_mse)
        model.train()
        
        #save the parameter for the lowest mae score for validation dataset
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'models/train_modelB_bestVal_lr2e5.pth')
            saved = True

            result = {'loss': loss,
                      'train_mae': train_mae,
                      'train_mse': train_mse,
                      'val_mae': val_mae,
                      'val_mse': val_mse}
            print("Best mae at: ")
            print(result)
                
        # print the results
        print('    ' + '-' * 20)
        print(' epoch: %d' %epoch)
        print('    [training mae %.3f validation mae %.3f]' % (train_mae, val_mae))
        print('    [training mse %.3f validation mse %.3f]' % (train_mse, val_mse))
        print('    ' + '-' * 20)
    torch.save(model.state_dict(), 'models/train_modelB_Val_lr2e5.pth')
    
    #saving traning and validation mase and mae
    with open('result_transfer/lr2e-5/modelB_maes_epoch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], mae_epoch))
    with open('result_transfer/lr2e-5/modelB_mses_epoch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], mse_epoch))       
    with open('result_transfer/lr2e-5/modelB_maes_batch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], mae_batch))
    with open('result_transfer/lr2e-5/modelB_mses_batch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], mse_batch))   
        
    with open('result_transfer/lr2e-5/modelB_val_maes_epoch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], val_mae_epoch))
    with open('result_transfer/lr2e-5/modelB_val_mses_epoch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], val_mse_epoch))       
    with open('result_transfer/lr2e-5/modelB_val_maes_batch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], val_mae_batch))
    with open('result_transfer/lr2e-5/modelB_val_mses_batch.csv','w') as new_file:
        write=csv.writer(new_file)
        write.writerows(map(lambda x: [x], val_mse_batch))       
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('SASNet inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
