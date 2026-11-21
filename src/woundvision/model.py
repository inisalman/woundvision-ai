import segmentation_models_pytorch as smp

def build_model():
    """
    Build WoundVision AI U-Net ResNet34.
    Output:
        6 Segmentation classes.
    """
    
    model = smp.Unet(
        encoder_name="resnet34",
        encode_weights="None",
        in_channels=3,
        classes=6,
        activation=None
    )

    return model
