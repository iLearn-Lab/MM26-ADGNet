import argparse
import time
import timeit
import torch.nn as nn
from torch.utils.data import DataLoader
from net import Net
from dataset import *
from metrics import *
import numpy as np
import os
import json as _json
from transformers import CLIPTextModel
from scipy.io import savemat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.tensorboard import SummaryWriter


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.textEncoder = CLIPTextModel.from_pretrained('./clip-vit-base-patch16')

    def forward(self, text, l_mask):
        with torch.no_grad():
            outputs = self.textEncoder(input_ids=text, attention_mask=l_mask)
            # [Batch, seq_len, 512] - last hidden state of all tokens
            sequence_output = outputs.last_hidden_state
            # [Batch, 512] - pooled [EOS] feature
            eot_output = outputs.pooler_output
            return sequence_output, eot_output


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


class Trainer(object):
    def __init__(self, opt):
        assert opt.mode == 'train' or opt.mode == 'test'

        self.mode = opt.mode
        seed = opt.seed

        def seed_worker(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        seed_worker(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        self.trainset = opt.trainset
        train_set = DataSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.trainset,
                                  mode='train', img_norm_cfg=opt.img_norm_cfg)
        self.train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers,
                                       batch_size=opt.batchSize, shuffle=True,
                                       worker_init_fn=seed_worker, generator=g)

        self.val_loaders = {}
        self.best_metric_mIoU = {}
        self.best_metric_nIoU = {}
        for dkey in args.testset:
            valset = DataSetLoader(dataset_dir=opt.dataset_dir, dataset_name=dkey,
                                   mode='val' if dkey == args.trainset else 'test',
                                   img_norm_cfg=opt.img_norm_cfg)
            self.val_loaders[dkey] = DataLoader(valset, 1, drop_last=False,
                                                num_workers=opt.num_workers, shuffle=False,
                                                worker_init_fn=seed_worker, generator=g)
            self.best_metric_mIoU[dkey] = 0.
            self.best_metric_nIoU[dkey] = 0.

        device = torch.device('cuda')
        self.device = device
        self.model = Net().to(device)
        self.text_encoder = TextEncoder().to(device)

        self.epoch_state = 0
        if opt.resume:
            if os.path.exists(opt.resume):
                print(f"==> Loading resume checkpoint: {opt.resume}")
                ckpt = torch.load(opt.resume)
                self.model.load_state_dict(ckpt['state_dict'])
                self.epoch_state = ckpt['epoch']
                # Adjust milestones: keep only future ones, offset by completed epochs
                opt.scheduler_settings['step'] = [
                    s - ckpt['epoch'] for s in opt.scheduler_settings['step']
                    if s > ckpt['epoch']
                ]
                if not opt.scheduler_settings['step']:
                    opt.scheduler_settings['step'] = [opt.epochs]
                print(f"==> Resumed successfully, starting from epoch {self.epoch_state}.")
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {opt.resume}")

        if opt.pretrained:
            if os.path.exists(opt.pretrained):
                print(f"==> Loading pretrained weights: {opt.pretrained}")
                ckpt = torch.load(opt.pretrained)
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                raise FileNotFoundError(f"Pretrained checkpoint not found: {opt.pretrained}")
        self.model = torch.nn.DataParallel(self.model)

        # Default optimizer / scheduler settings (only applied when not provided via CLI)
        if opt.optimizer_settings is None or opt.scheduler_name is None or opt.scheduler_settings is None:
            if opt.optimizer_name == 'Adam':
                if opt.optimizer_settings is None:
                    opt.optimizer_settings = {'lr': 5e-4}
                if opt.scheduler_name is None:
                    opt.scheduler_name = 'MultiStepLR'
                if opt.scheduler_settings is None:
                    opt.scheduler_settings = {'epochs': opt.epochs, 'step': [200, 400], 'gamma': 0.1}
            elif opt.optimizer_name == 'Adagrad':
                if opt.optimizer_settings is None:
                    opt.optimizer_settings = {'lr': 0.05}
                if opt.scheduler_name is None:
                    opt.scheduler_name = 'CosineAnnealingLR'
                if opt.scheduler_settings is None:
                    opt.scheduler_settings = {'epochs': opt.epochs, 'min_lr': 1e-5}
            else:
                raise NotImplementedError

        self.optimizer, self.scheduler = get_optimizer(
            self.model, opt.optimizer_name, opt.scheduler_name,
            opt.optimizer_settings, opt.scheduler_settings)

        self.savename = opt.name + '-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        if self.mode == 'train':
            self.savename += '-' + opt.trainset
        else:
            self.savename += '-' + '/'.join(args.testset)

        if self.mode == 'train':
            self.writer = SummaryWriter(f'./tf-logs/{self.savename}')
            self.save_folder = f'./Weights/{self.savename}'
            if not osp.exists(self.save_folder):
                os.makedirs(self.save_folder)

        self.opt = opt

    def train(self, epoch):
        tic = timeit.default_timer()
        self.model.train()
        train_loss = 0

        for data, mask, text_ids_fg, l_mask_fg, text_ids_bg, l_mask_bg in self.train_loader:
            data = data.to(self.device)
            labels = mask.to(self.device)
            l_mask_bg = l_mask_bg.squeeze(1).to(self.device)
            l_mask_fg = l_mask_fg.squeeze(1).to(self.device)
            text_ids_bg = text_ids_bg.to(self.device)
            text_ids_fg = text_ids_fg.to(self.device)
            _, text_eot = self.text_encoder(text_ids_bg, l_mask_bg)
            text_sequence, _ = self.text_encoder(text_ids_fg, l_mask_fg)

            pred = self.model(data, text_sequence, l_mask_fg, text_eot, l_mask_bg)
            loss = self.model.module.loss(pred, labels)
            train_loss += loss.detach().cpu()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        train_loss /= len(self.train_loader)

        self.writer.add_scalar('train_loss', train_loss, epoch)
        toc = timeit.default_timer()
        print('Train epoch {:03d} on {}, loss: {:.4f}, time {}m{}s.'.format(
            epoch, self.trainset, train_loss, int((toc - tic) // 60), int(toc - tic) % 60))

    def test(self, epoch):
        self.model.eval()

        for dkey in self.val_loaders.keys():
            tic = timeit.default_timer()

            fn_mIoU = mIoU()
            fn_PD_FA = PD_FA()

            with torch.no_grad():
                for data, mask, size, _, text_ids_fg, l_mask_fg, text_ids_bg, l_mask_bg in self.val_loaders[dkey]:
                    data = data.to(self.device)
                    l_mask_bg = l_mask_bg.squeeze(1).to(self.device)
                    l_mask_fg = l_mask_fg.squeeze(1).to(self.device)
                    text_ids_bg = text_ids_bg.to(self.device)
                    text_ids_fg = text_ids_fg.to(self.device)
                    _, text_eot = self.text_encoder(text_ids_bg, l_mask_bg)
                    text_sequence, _ = self.text_encoder(text_ids_fg, l_mask_fg)

                    pred = self.model(data, text_sequence, l_mask_fg, text_eot, l_mask_bg)
                    pred = pred[:, :, :size[0], :size[1]]
                    mask_crop = mask[:, :, :size[0], :size[1]]
                    fn_mIoU.update((pred > self.opt.threshold).cpu(), mask_crop)
                    fn_PD_FA.update((pred[0, 0] > self.opt.threshold).cpu(), mask_crop[0, 0], size)

            eval_pixAcc, eval_mIoU = fn_mIoU.get()
            eval_Pd, eval_Fa = fn_PD_FA.get()
            _, eval_nIoU = fn_mIoU.get_single()

            all_log_path = osp.join(self.save_folder, f'metrics_all_epochs_{dkey}.log')
            with open(all_log_path, 'a') as f:
                f.write('{} - {:04d}\t - pixAcc {:.4f}\t - mIoU {:.4f}\t nIoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'
                        .format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                epoch, eval_pixAcc * 1e2, eval_mIoU * 1e2, eval_nIoU * 1e2,
                                eval_Pd * 1e2, eval_Fa * 1e6))

            if self.mode == 'train':
                self.writer.add_scalar(f'eval-{dkey}-pixAcc', eval_pixAcc * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-mIoU', eval_mIoU * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-nIoU', eval_nIoU * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-Pd', eval_Pd * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-Fa', eval_Fa * 1e6, epoch)

                if eval_mIoU > self.best_metric_mIoU[dkey]:
                    self.best_metric_mIoU[dkey] = eval_mIoU
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'eval_pixAcc': eval_pixAcc,
                        'eval_mIoU': eval_mIoU,
                        'eval_nIoU': eval_nIoU,
                        'eval_Pd': eval_Pd,
                        'eval_Fa': eval_Fa},
                        osp.join(self.save_folder, f'best_mIoU_on_{dkey}.pth.tar'))
                    with open(osp.join(self.save_folder, f'metrics_mIoU_on_{dkey}.log'), 'a') as f:
                        f.write('{} - {:04d}\t - pixAcc {:.4f}\t - mIoU {:.4f}\t nIoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'
                                .format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                        epoch, eval_pixAcc * 1e2, eval_mIoU * 1e2, eval_nIoU * 1e2,
                                        eval_Pd * 1e2, eval_Fa * 1e6))

                if eval_nIoU > self.best_metric_nIoU[dkey]:
                    self.best_metric_nIoU[dkey] = eval_nIoU
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'eval_pixAcc': eval_pixAcc,
                        'eval_mIoU': eval_mIoU,
                        'eval_nIoU': eval_nIoU,
                        'eval_Pd': eval_Pd,
                        'eval_Fa': eval_Fa},
                        osp.join(self.save_folder, f'best_nIoU_on_{dkey}.pth.tar'))
                    with open(osp.join(self.save_folder, f'metrics_nIoU_on_{dkey}.log'), 'a') as f:
                        f.write('{} - {:04d}\t - pixAcc {:.4f}\t - mIoU {:.4f}\t nIoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'
                                .format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                        epoch, eval_pixAcc * 1e2, eval_mIoU * 1e2, eval_nIoU * 1e2,
                                        eval_Pd * 1e2, eval_Fa * 1e6))

            toc = timeit.default_timer()
            print('Eval on {}, time {}m{}s.'.format(dkey, int((toc - tic) // 60), int(toc - tic) % 60))
            print('pixAcc: {:.4f}  mIoU: {:.4f}  Best_mIoU: {:.4f}'.format(
                eval_pixAcc * 1e2, eval_mIoU * 1e2, self.best_metric_mIoU[dkey] * 1e2))
            print('nIoU: {:.4f}  Best_nIoU: {:.4f}'.format(
                eval_nIoU * 1e2, self.best_metric_nIoU[dkey] * 1e2))
            print('Pd: {:.4f}  Fa: {:.4f} (x1e-6)'.format(eval_Pd * 1e2, eval_Fa * 1e6))
            print('')

    def inference(self, save_output=True):
        ToImg = transforms.ToPILImage()

        ckpt = torch.load(self.opt.ckpt, map_location=self.device)
        self.model.module.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        for dkey in self.val_loaders.keys():
            tic = timeit.default_timer()

            fn_mIoU = mIoU()
            fn_PD_FA = PD_FA()

            output_path = f'./outputs/{self.savename}/pngs'
            mat_output_path = f'./outputs/{self.savename}/mats'

            if save_output:
                if not osp.exists(output_path):
                    os.makedirs(output_path)
                if not osp.exists(mat_output_path):
                    os.makedirs(mat_output_path)

            with torch.no_grad():
                for data, mask, size, filename, text_ids_fg, l_mask_fg, text_ids_bg, l_mask_bg in self.val_loaders[dkey]:
                    data = data.to(self.device)
                    l_mask_bg = l_mask_bg.squeeze(1).to(self.device)
                    l_mask_fg = l_mask_fg.squeeze(1).to(self.device)
                    text_ids_bg = text_ids_bg.to(self.device)
                    text_ids_fg = text_ids_fg.to(self.device)

                    _, text_eot = self.text_encoder(text_ids_bg, l_mask_bg)
                    text_sequence, _ = self.text_encoder(text_ids_fg, l_mask_fg)

                    pred = self.model(data, text_sequence, l_mask_fg, text_eot, l_mask_bg)
                    pred = pred[:, :, :size[0], :size[1]]
                    mask_crop = mask[:, :, :size[0], :size[1]]

                    fn_mIoU.update((pred > self.opt.threshold).cpu(), mask_crop)
                    fn_PD_FA.update((pred[0, 0] > self.opt.threshold).cpu(), mask_crop[0, 0], size)

                    if save_output:
                        for j_ in range(pred.shape[0]):
                            j_pred_bin = (pred[j_].detach().cpu() > self.opt.threshold).float()
                            ToImg(j_pred_bin).save(osp.join(output_path, filename[j_] + '.png'))
                            j_pred_np = pred[j_].detach().cpu().numpy().squeeze()
                            savemat(osp.join(mat_output_path, filename[j_] + '.mat'), {'predict_map': j_pred_np})

            eval_pixAcc, eval_mIoU = fn_mIoU.get()
            eval_Pd, eval_Fa = fn_PD_FA.get()
            _, eval_nIoU = fn_mIoU.get_single()
            toc = timeit.default_timer()

            print('=========================')
            print('Inference on {}, time {}m{}s'.format(dkey, int((toc - tic) // 60), int(toc - tic) % 60))
            print('pixAcc: {:.4f}  mIoU: {:.4f}  nIoU: {:.4f}'.format(eval_pixAcc * 1e2, eval_mIoU * 1e2, eval_nIoU * 1e2))
            print('Pd: {:.4f}  Fa: {:.4f} (x1e-6)'.format(eval_Pd * 1e2, eval_Fa * 1e6))
            print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch ADGNet train")
    parser.add_argument("--name", default='ADGNet', help="experiment description")

    parser.add_argument("--trainset", default='IRSTD-1K',
                        help="dataset: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'")
    parser.add_argument("--testset", default='IRSTD-1K',
                        help="dataset: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'")

    parser.add_argument("--img_norm_cfg", default=None, type=dict,
                        help="custom img_norm_cfg dict, default=None (auto-detect per dataset)")
    parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                        help="custom mean for img_norm_cfg")
    parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                        help="custom std for img_norm_cfg")

    parser.add_argument("--dataset_dir", default='/root/autodl-tmp/ADGNet/datasets',
                        type=str, help="path to dataset directory")

    parser.add_argument("--batchSize", type=int, default=16, help="training batch size")

    parser.add_argument("--resume", default=None, type=str, help="resume checkpoint path")
    parser.add_argument("--pretrained", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--ckpt", default=None, type=str, help="inference checkpoint path (--mode test)")

    parser.add_argument("--epochs", type=int, default=600, help="number of training epochs")
    parser.add_argument("--optimizer_name", default='Adam', type=str, help="Adam, Adagrad, SGD")
    parser.add_argument("--optimizer_settings", default=None, type=str, help="optimizer settings (JSON string)")
    parser.add_argument("--scheduler_name", default=None, type=str, help="MultiStepLR, CosineAnnealingLR")
    parser.add_argument("--scheduler_settings", default=None, type=str, help="scheduler settings (JSON string)")

    parser.add_argument("--num_workers", type=int, default=8, help="data loader workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="binary threshold for inference")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--mode", type=str, default='train', help="train or test")
    parser.add_argument("--test_freq", type=int, default=1, help="evaluation frequency (epochs)")
    args = parser.parse_args()

    args.testset = args.testset.split('/')

    # Parse JSON string arguments
    for attr in ('optimizer_settings', 'scheduler_settings'):
        val = getattr(args, attr)
        if isinstance(val, str):
            setattr(args, attr, _json.loads(val))

    if args.img_norm_cfg_mean is not None and args.img_norm_cfg_std is not None:
        args.img_norm_cfg = dict()
        args.img_norm_cfg['mean'] = args.img_norm_cfg_mean
        args.img_norm_cfg['std'] = args.img_norm_cfg_std

    print('///////////////////////////////////////////////////////')
    print(args)

    trainer = Trainer(args)

    if trainer.mode == 'train':
        print('\n========== Training ==========')
        for epoch in range(trainer.epoch_state, args.epochs):
            trainer.train(epoch)
            if (epoch + 1) % args.test_freq == 0:
                print('-----------------------')
                trainer.test(epoch)

    if args.ckpt:
        print('\n========== Inference ==========')
        trainer.inference()
