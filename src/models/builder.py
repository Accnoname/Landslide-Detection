try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

def build_model(arch="unet", encoder_name="resnet34", in_channels=14, num_classes=1):
    """
    Khởi tạo mô hình phân đoạn ngữ nghĩa ảnh vệ tinh đa kênh.
    
    Hỗ trợ 4 kiến trúc cốt lõi:
    - unet: ResU-Net (U-Net với ResNet34 backbone)
    - fpn: Feature Pyramid Network
    - unetplusplus: Nested U-Net (UNet++)
    - deeplabv3plus: DeepLabV3+ với Atrous Spatial Pyramid Pooling (ASPP)
    """
    if not HAS_SMP:
        raise ImportError("Vui lòng cài đặt segmentation_models_pytorch: pip install segmentation-models-pytorch")

    arch = arch.lower()
    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,      # Huấn luyện từ đầu cho 14 kênh viễn thám
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    elif arch == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    elif arch == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    elif arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Không hỗ trợ kiến trúc '{arch}'. Các lựa chọn: ['unet', 'fpn', 'unetplusplus', 'deeplabv3plus']")
