

##this is for seing how i can retrieve the savepoint of the model

from unet3d import utils
from unet3d.model import UNet3D

if __name__ == "__main__":
    
    model = UNet3D(
        in_channels=1,
        out_channels=13,
        f_maps=(32, 64, 128, 256, 512),
        layer_order="cgr",
        num_groups=8,
        final_sigmoid=False,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="deconv",
        num_levels=5,
        dropout_prob=0.0,
        is_segmentation=True,
        is3d=True,
    )
    
    
    # Load the model
    state = utils.load_checkpoint(
       "./checkpoints",
        model,
    )
    
    print(state)
    
    print(state.keys())