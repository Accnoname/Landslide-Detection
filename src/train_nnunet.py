import os
import sys
import argparse
import torch

# 1. Thiet lap bien moi truong bat buoc cho nnU-Net v2
os.environ["nnUNet_raw"] = "D:/Detect/data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "D:/Detect/data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "D:/Detect/data/nnUNet_results"
os.environ["nnUNet_n_proc_DA"] = "0"

import nnunetv2
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_Landslide(nnUNetTrainer):
    """
    Phien ban nnUNetTrainer tuy bien cho anh vien tham da pho 14 kenh:
    - Chay o che do FP32 (khong dung FP16 de chong tran so NaN voi kenh DEM/Slope)
    - Gradient clipping = 12 giup hoi tu cuc ky on dinh
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        plans["continue_training"] = False
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.grad_scaler = None # Vo hieu hoa GradScaler FP16

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Chay thuan FP32 de bao toan do chinh xac tuyet doi cho 14 kenh
        with dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=None)
        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

def main():
    parser = argparse.ArgumentParser(description="Chay Huan Luyen nnU-Net v2 cho Phat Hien Sat Lo")
    parser.add_argument("--epochs", type=int, default=5, help="So epochs huan luyen")
    parser.add_argument("--iters_per_epoch", type=int, default=20, help="So iterations moi epoch")
    parser.add_argument("--val_iters", type=int, default=10, help="So iterations validation")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"[nnU-Net v2] KHOI CHAY HUAN LUYEN CHINH THUC GIAI DOAN 2 ({args.epochs} EPOCHS)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Thiet bi: {device} | Card: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    from batchgenerators.utilities.file_and_folder_operations import load_json
    plans = load_json("D:/Detect/data/nnUNet_preprocessed/Dataset001_Landslide/nnUNetPlans.json")
    dataset_json = load_json("D:/Detect/data/nnUNet_raw/Dataset001_Landslide/dataset.json")

    print("[*] Dang khoi tao nnUNetTrainer_Landslide FP32 (Anti-NaN)...")
    trainer = nnUNetTrainer_Landslide(
        plans=plans,
        configuration="2d",
        fold=0,
        dataset_json=dataset_json,
        device=device
    )

    trainer.num_epochs = args.epochs
    trainer.num_iterations_per_epoch = args.iters_per_epoch
    trainer.num_val_iterations_per_epoch = args.val_iters

    print(f"[CONFIG] Dynamic 2D PlainConvUNet (6 Stages: 32 -> 64 -> 128 -> 256 -> 512 -> 512)")
    print(f"[CONFIG] Input: 14 Channels | Loss: Compound DC_and_CE_loss (Deep Supervision)")
    print(f"[CONFIG] So Epochs: {trainer.num_epochs} | Iters/Epoch: {trainer.num_iterations_per_epoch}")
    print("="*80 + "\n")

    print("[RUN] Dang huan luyen tren GPU...")
    trainer.run_training()

    # Copy ket qua ra thu muc results/plots
    import shutil
    res_dir = "D:/Detect/data/nnUNet_results/Dataset001_Landslide/nnUNetTrainer_Landslide__nnUNetPlans__2d/fold_0"
    if os.path.exists(res_dir):
        prog_file = os.path.join(res_dir, "progress.png")
        if os.path.exists(prog_file):
            shutil.copy(prog_file, "D:/Detect/results/plots/nnunet_progress.png")
            shutil.copy(prog_file, "C:/Users/admin/.gemini/antigravity-ide/brain/a14a4172-e21d-45af-8beb-f0084268ed28/nnunet_progress.png")

    print("\n" + "="*80)
    print(f"[THANH CONG] DA HOAN TAT {args.epochs} EPOCHS HUAN LUYEN nnU-Net v2!")
    print(f"[*] Thu muc ket qua: {res_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
