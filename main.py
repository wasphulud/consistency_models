from training.train import initialize_inputs

RUN = True
PATH_ROOT = "/Users/aimans/Storage/consistency_models/"
if __name__ == "__main__":
    # timesteps_s = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # input_tensor_s = torch.randn(10, 3, 64, 64)
    from data.dataloader import load_data
    from models.unet import Unet

    loader = load_data("/Users/aimans/Storage/imagenet-mini/train", 10, 64, True, True)

    unet = Unet(3, 3, 128, [1, 2, 3, 4], 3, 0.5, [2, 4, 8], num_classes=0)
    initialize_inputs(
        loader, unet, "/Users/aimans/Storage/consistency_models/edm_imagenet64_ema.pt"
    )
