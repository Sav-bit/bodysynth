

##this is for seing how i can retrieve the savepoint of the model

from unet3d import utils
from unet3d.model import UNet3D
import nibabel as nib
from nibabel.orientations import axcodes2ornt
from nibabel.orientations import ornt_transform


def get_orientation(nii: nib.Nifti1Image) -> tuple[str, str, str]:
    """Gets the orientation of a nifti image."""
    orientation = nib.aff2axcodes(nii.affine)
    return orientation

def reorient(
    nii: nib.Nifti1Image,
    orientation: str | tuple[str, str, str] = "RAS",
) -> nib.Nifti1Image:
    """Reorients a nifti image to specified orientation. Orientation string or tuple
    must consist of "R" or "L", "A" or "P", and "I" or "S" in any order."""
    orig_ornt = nib.io_orientation(nii.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    reoriented_nii = nii.as_reoriented(transform)
    return reoriented_nii


if __name__ == "__main__":
    
    # model = UNet3D(
    #     in_channels=1,
    #     out_channels=13,
    #     f_maps=(32, 64, 128, 256, 512),
    #     layer_order="cgr",
    #     num_groups=8,
    #     final_sigmoid=False,
    #     conv_kernel_size=3,
    #     pool_kernel_size=2,
    #     conv_padding=1,
    #     conv_upscale=2,
    #     upsample="deconv",
    #     num_levels=5,
    #     dropout_prob=0.0,
    #     is_segmentation=True,
    #     is3d=True,
    # )
    
    
    # # Load the model
    # state = utils.load_checkpoint(
    #    "./checkpoints/last_checkpoint.pytorch",
    #     model,
    # )
    
    # print(state)
    
    # print(state.keys())
    
    img="/Users/sav/Documents/Progetti DTU/medical-segmentator/ErnieExtended/m2m_ernie_extended/T1.nii.gz"
    seg="/Users/sav/Documents/Progetti DTU/medical-segmentator/ernie_less_dim.nii.gz"
    
    # Load the image and segmentation
    img_nifti = nib.load(img)
    seg_nifti = nib.load(seg)
    
    img_data = img_nifti.get_fdata()
    seg_data = seg_nifti.get_fdata()
    print(f"Image shape: {img_data.shape}, Segmentation shape: {seg_data.shape}")
    
    img_orientation = get_orientation(img_nifti)
    seg_orientation = get_orientation(seg_nifti)
    print(f"Image orientation: {img_orientation}, Segmentation orientation: {seg_orientation}")
    
    #reorient the image and segmentation to RAS orientation
    img_reoriented = reorient(img_nifti, "RAS")
    seg_reoriented = reorient(seg_nifti, "RAS")
    
    print(f"Reoriented Image orientation: {get_orientation(img_reoriented)}")
    print(f"Reoriented Segmentation orientation: {get_orientation(seg_reoriented)}")
    
    