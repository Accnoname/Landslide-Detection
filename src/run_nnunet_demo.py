import os
import sys
import torch

# 1. Thiet lap bien moi truong bat buoc cho nnU-Net v2
os.environ["nnUNet_raw"] = "D:/Detect/data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "D:/Detect/data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "D:/Detect/data/nnUNet_results"

import nnunetv2
from nnunetv2.run.run_training import get_trainer_from_args

def main():
    print("\n" + "="*80)
    print("[GIAI DOAN 2] CHAY THU NGHIEM PIPELINE nnU-Net v2 (Y TUONG COT LOI CUA THAY)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Thiet bi: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[PATH]   nnUNet_raw:          {os.environ['nnUNet_raw']}")
    print(f"[PATH]   nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"[PATH]   nnUNet_results:      {os.environ['nnUNet_results']}")

    # 2. Khoi tao nnUNetTrainer cho Dataset 1 (2D configuration, Fold 0)
    print("\n[*] Dang khoi tao nnUNetTrainer chuan quoc te...")
    trainer = get_trainer_from_args(
        dataset_name_or_id="1",
        configuration="2d",
        fold=0,
        trainer_name="nnUNetTrainer",
        plans_identifier="nnUNetPlans",
        device=device
    )

    # 3. Cau hinh che do CHAY THU (Quick Test Demo):
    # nnU-Net mac dinh chay 1000 epochs (mat hang ngay troi).
    # Ta cau hinh 2 epochs, moi epoch 10 iterations de ban xem truc tiep co che hoat dong ngay lap tuc!
    trainer.num_epochs = 2
    trainer.num_iterations_per_epoch = 10
    trainer.num_val_iterations_per_epoch = 5

    print(f"[CONFIG] Kien truc mang:          PlainConvUNet (Dynamic 6 stages, 14 channels in)")
    print(f"[CONFIG] Ham mat mat (Loss):      Compound DC_and_CE_loss (Dice + CrossEntropy)")
    print(f"[CONFIG] Toi uu hoa (Optimizer):  SGD with Nesterov Momentum (lr=0.01, PolyLR)")
    print(f"[CONFIG] Che do chay thu:         {trainer.num_epochs} Epochs ({trainer.num_iterations_per_epoch} iters/epoch)")
    print("="*80 + "\n")

    # 4. Khoi chay huan luyen
    print("[RUN] Bat dau chu trinh huan luyen nnU-Net v2...")
    trainer.run_training()

    print("\n" + "="*80)
    print("[THANH CONG] DA HOAN TAT CHAY THU NGHIEM GIAI DOAN 2 VOI nnU-Net v2!")
    print(f"[*] Checkpoint va log luu tai: D:/Detect/data/nnUNet_results/Dataset001_Landslide/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
