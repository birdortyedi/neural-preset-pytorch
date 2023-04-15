import os
import yaml
import torch
import glog as log
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from dncm import DNCM, Encoder
from dataset import MSCOCO
from dataset import Compose, ToTensor, RandomCropThreeInstances, RandomHorizontalFlipThreeInstances


class Trainer:
    def __init__(self, cfg) -> None:
        self._setup(cfg)
        self._init_parameters()
        
        self.wandb = wandb
        self.wandb.init(
            project=self.PROJECT_NAME,
            resume=self.INIT_FROM is not None, 
            notes=str(self.LOG_DIR), 
            config=self.cfg, 
            entity=self.ENTITY
        )
        
        self.transform = Compose([
            RandomCropThreeInstances((self.IMG_SIZE, self.IMG_SIZE)),
            RandomHorizontalFlipThreeInstances(),
            ToTensor()
        ])

        self.dataset = MSCOCO(root=self.DATASET_ROOT, transform=self.transform, is_train=True)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE, num_workers=self.NUM_WORKERS)
        self.to_pil = transforms.ToPILImage()
        
        self.nDNCM = DNCM(self.k)
        self.sDNCM = DNCM(self.k)
        self.encoder = Encoder(self.sz, self.k)

        self.optimizer = torch.optim.Adam(
            list(self.nDNCM.parameters()) + list(self.sDNCM.parameters()) + list(self.encoder.parameters()),
            lr=self.LR, betas=self.BETAS
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.SCHEDULER_STEP, gamma=self.SCHEDULER_GAMMA)

        self.current_epoch = 0
        if self.INIT_FROM is not None and self.INIT_FROM != "":
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoints(self.INIT_FROM)
        self.check_and_use_multi_gpu()
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.l2_loss = torch.nn.MSELoss().cuda()

    def _setup(self, cfg):
        with open(cfg, 'r') as stream:
            self.cfg = yaml.safe_load(stream)
    
    def _init_parameters(self):
        self.k = int(self.cfg["k"])
        self.sz = int(self.cfg["sz"])
        self.LR = float(self.cfg["LR"])
        self.BETAS = (self.cfg["BETA1"], self.cfg["BETA2"])
        self.NUM_GPU = int(self.cfg["NUM_GPU"])
        self.DATASET_ROOT = Path(self.cfg["DATASET_ROOT"])
        self.IMG_SIZE = int(self.cfg["IMG_SIZE"])
        self.BATCH_SIZE = int(self.cfg["BATCH_SIZE"])
        self.EPOCHS = int(self.cfg["EPOCHS"])
        self.LAMBDA = float(self.cfg["LAMBDA"])
        self.SCHEDULER_STEP = int(self.cfg["SCHEDULER_STEP"])
        self.SCHEDULER_GAMMA = float(self.cfg["SCHEDULER_GAMMA"])
        self.VISUALIZE_STEP = int(self.cfg["VISUALIZE_STEP"])
        self.SHUFFLE = bool(self.cfg["SHUFFLE"])
        self.NUM_WORKERS = int(self.cfg["NUM_WORKERS"])
        self.CKPT_DIR = Path(self.cfg["CKPT_DIR"])
        self.INIT_FROM = self.cfg["INIT_FROM"]
        self.PROJECT_NAME = self.cfg["PROJECT_NAME"]
        self.LOG_DIR = Path(self.cfg["LOG_DIR"])
        self.ENTITY = self.cfg["ENTITY"]
                
    def __call__(self):
        self.run()
    
    def run(self):
        for e in range(self.EPOCHS):
            log.info(f"Epoch {e+1}/{self.EPOCHS}")
            for step, (I, I_i, I_j) in enumerate(tqdm(self.image_loader, total=len(self.image_loader))):
                self.optimizer.zero_grad()

                I = I.float().cuda()
                I_i = I_i.float().cuda()
                I_j = I_j.float().cuda()
                
                # from Figure 4 in https://arxiv.org/pdf/2303.13511.pdf
                d_i, r_i = self.encoder(I_i)
                d_j, r_j = self.encoder(I_j)
                Z_i = self.nDNCM(I_i, d_i)
                Z_j = self.nDNCM(I_j, d_j)
                Y_i = self.sDNCM(Z_j, r_i)
                Y_j = self.sDNCM(Z_i, r_j)
                consistency_loss = self.l2_loss(Z_i, Z_j)
                reconstruction_loss = self.l1_loss(Y_i, I_i) + self.l1_loss(Y_j, I_j)
                final_loss = reconstruction_loss + self.LAMBDA * consistency_loss
                
                final_loss.backward()
                self.optimizer.step()
                self.wandb.log({
                    "consistency_loss": consistency_loss.item(),
                    "reconstruction_loss": reconstruction_loss.item()
                }, commit=False)
                if step % self.VISUALIZE_STEP == 0 and step != 0:
                    self.visualize(I, I_i, I_j, Y_i, Y_j, Z_i, Z_j)
                else:
                    self.wandb.log({})
            self.scheduler.step()
            self.do_checkpoint()

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.NUM_GPU > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs...")
            self.nDNCM = torch.nn.DataParallel(self.nDNCM).cuda()
            self.sDNCM = torch.nn.DataParallel(self.sDNCM).cuda()
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        else:
            log.info(f"GPU ID: {torch.cuda.current_device()}")
            self.nDNCM = self.nDNCM.cuda()
            self.sDNCM = self.sDNCM.cuda()
            self.encoder = self.encoder.cuda()

    def do_checkpoint(self):
        os.makedirs(str(self.CKPT_DIR), exist_ok=True)
        checkpoint = {
            'epoch': self.current_epoch,
            'nDCNM': self.nDNCM.module.state_dict() if isinstance(self.nDNCM, torch.nn.DataParallel) else self.nDNCM.state_dict(),
            'sDCNM': self.sDNCM.module.state_dict() if isinstance(self.sDNCM, torch.nn.DataParallel) else self.sDNCM.state_dict(),
            'encoder': self.encoder.module.state_dict() if isinstance(self.encoder, torch.nn.DataParallel) else self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, str(self.CKPT_DIR / "latest_ckpt.pth"))
    
    def load_checkpoints(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.nDNCM.load_state_dict(checkpoints["nDCNM"])
        self.sDNCM.load_state_dict(checkpoints["sDCNM"])
        self.encoder.load_state_dict(checkpoints["encoder"])
        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.init_epoch = checkpoints["epoch"]
        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def visualize(self, I, I_i, I_j, Y_i, Y_j, Z_i, Z_j):
        idx = 0
        self.wandb.log({"examples": [
            self.wandb.Image(self.to_pil(I[idx].cpu()), caption="I"),
            self.wandb.Image(self.to_pil(I_i[idx].cpu()), caption="I_i"),
            self.wandb.Image(self.to_pil(I_j[idx].cpu()), caption="I_j"),
            self.wandb.Image(self.to_pil(torch.clamp(Y_i, min=0., max=1.)[idx].cpu()), caption="Y_i"),
            self.wandb.Image(self.to_pil(torch.clamp(Y_j, min=0., max=1.)[idx].cpu()), caption="Y_j"),
            self.wandb.Image(self.to_pil(torch.clamp(Z_i, min=0., max=1.)[idx].cpu()), caption="Z_i"),
            self.wandb.Image(self.to_pil(torch.clamp(Z_j, min=0., max=1.)[idx].cpu()), caption="Z_j")
        ]}, commit=False)
        self.wandb.log({})