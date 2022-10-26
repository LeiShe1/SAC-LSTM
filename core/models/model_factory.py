

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import predict
from core.layers.Discriminator import Discriminatorr

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        
        
        #导入模型结构
        networks_map = {
            'convlstm':predict.ConvLSTM,
            'predrnn':predict.PredRNN,
            'predrnn_plus': predict.PredRNN_Plus,
            'interact_convlstm': predict.InteractionConvLSTM,
            'interact_predrnn':predict.InteractionPredRNN,
            'interact_predrnn_plus':predict.InteractionPredRNN_Plus,
            'cst_predrnn':predict.CST_PredRNN,
            'sst_predrnn': predict.SST_PredRNN,
            'dst_predrnn':predict.DST_PredRNN,
            'interact_dst_predrnn': predict.InteractionDST_PredRNN,
        }

        if configs.model_name in networks_map:

            Network = networks_map[configs.model_name]
            # self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            self.Discriminator = Discriminatorr(self.patch_height, self.patch_width, self.patch_channel,
                                           self.configs.D_num_hidden).to(configs.device)
            self.network = nn.DataParallel(self.network)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        if self.configs.is_parallel:
            self.network = nn.DataParallel(self.network)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.optimizer_D = Adam(self.Discriminator.parameters(), lr=configs.lr_d)
        
        self.MSE_criterion = nn.MSELoss(size_average=False)
        self.D_criterion = nn.BCELoss()
        self.L1_loss = nn.L1Loss(size_average=False)


    def save(self,ite = None):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_pm.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)
        
        stats = {}
        stats['net_param'] = self.Discriminator.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        torch.save(stats, checkpoint_path)
        print("save discriminator model to %s" % checkpoint_path)

    def load(self):
        print('model has been loaded:')
        checkpoint_path_pm = os.path.join(self.configs.save_dir, 'model_pm.ckpt')
        stats = torch.load(checkpoint_path_pm)
        self.network.load_state_dict(stats['net_param'],False)
        
        print('load discriminator model:')
        checkpoint_path_d = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        stats = torch.load(checkpoint_path_d)
        self.Discriminator.load_state_dict(stats['net_param'])
        
        

    def train(self, frames, mask):
        # "no-gan"
        # self.network.train()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        # self.optimizer.zero_grad()
        # next_frames = self.network(frames_tensor, mask_tensor)
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])+\
        #        self.L1_loss(next_frames, frames_tensor[:, 1:])
        # loss.backward()
        # self.optimizer.step()
        # return loss.detach().cpu().numpy()
        
        "gan"
        self.network.train()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        ground_truth = frames_tensor[:, 1:]
        
        next_frames = next_frames.permute(0, 1, 4, 2, 3)
        ground_truth = ground_truth.permute(0, 1, 4, 2, 3)
        
        batch_size = next_frames.shape[0]
        zeros_label = torch.zeros(batch_size).cuda()
        ones_label = torch.ones(batch_size).cuda()
        
        # train D
        self.Discriminator.zero_grad()
        d_gen, _ = self.Discriminator(next_frames.detach())
        d_gt, _ = self.Discriminator(ground_truth)
        D_loss = self.D_criterion(d_gen, zeros_label) + self.D_criterion(d_gt, ones_label)
        D_loss.backward(retain_graph=True)
        self.optimizer_D.step()
        
        self.optimizer.zero_grad()
        d_gen_pre, features_gen = self.Discriminator(next_frames)
        _, features_gt = self.Discriminator(ground_truth)
        
        loss_l1 = self.L1_loss(next_frames, ground_truth)
        loss_l2 = self.MSE_criterion(next_frames, ground_truth)
        gen_D_loss = self.D_criterion(d_gen_pre, ones_label)
        loss_features = self.MSE_criterion(features_gen, features_gt)
        loss = loss_l1 + loss_l2 + 0.01*loss_features + 0.001*gen_D_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
      

    def test(self, frames, mask):
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).cuda()
        mask_tensor = torch.FloatTensor(mask).cuda()
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) +\
               self.L1_loss(next_frames,frames_tensor[:,1:])
               # + 0.02 * self.SSIM_criterion(next_frames, frames_tensor[:, 1:])

        return next_frames.detach().cpu().numpy(),loss.detach().cpu().numpy()

