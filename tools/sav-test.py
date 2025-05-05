import os
import torch

import brainsynth
import nibabel as nib
import numpy as np

from brainsynth.constants import ImageSettings


"""
This script is used to test the Sav synthesizer from brainsynth.
"""

def save_nifti_images(img, result, suffix):
    #get the affine
    affine = img.affine
    print(f"result keys: {result.keys()}")
    os.makedirs("new-output", exist_ok=True)
    save_in_dir = lambda x: os.path.join("new-output", x)
    #save the result
    toSave = nib.Nifti1Image(result["image"][0].cpu().numpy(), affine, img.header)
    nib.save(toSave, save_in_dir(f"{suffix}_image_sav.nii.gz"))
    #save the t1w image
    print(f"result len: {len(result["seg"])}")
    # Cast result["seg"] to int64 before computing argmax
    seg_int = result["seg"].to(torch.int64)
    print(f"seg_int dtype: {seg_int.dtype}")  # Check new dtype
    combined_seg = torch.argmax(seg_int, dim=0)
    toSave = nib.Nifti1Image(combined_seg.cpu().numpy(), affine, img.header)
    nib.save(toSave, save_in_dir(f"{suffix}_seg_sav.nii.gz"))
    print("Result saved for sav synthesizer")

if __name__ == "__main__":

    #For now jsut check the pythorch version
    print(torch.__version__)
    #Check if MPS is available
    if torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
    else:
        print("MPS is not available")
        #Check if CUDA is available
        if torch.cuda.is_available():
            print("CUDA is available")
            device = torch.device("cuda")
        else:
            print("CUDA is not available")
            device = torch.device("cpu")
            
    device = torch.device("cpu")

            
    print(f"Using device: {device}")
    

    #load ernie segmentation
    ernie_path = "/Users/sav/Documents/Progetti DTU/medical-segmentator/ernie_less_dim.nii.gz"
    
    #Load it with nibabel
    img = nib.load(ernie_path)
    data = img.get_fdata()
    
    next_even_number = lambda x: x if x % 2 == 0 else x + 1
    
    out_size = [next_even_number(data.shape[0]), next_even_number(data.shape[1]), next_even_number(data.shape[2])]
    
    print(f"out_size: {out_size}")
    
    #get all the labels of the image
    segmentation_labels = np.unique(data)
    print(f"labels: {segmentation_labels}")
    
    #for now let's overwite it to 40 x 40 x 40
    out_size = [200, 200, 200]
    
    # out_center_str = "image"
    out_center_str = "random"
    
    IMAGE = ImageSettings()
    print("Segmentation labels:", getattr(IMAGE.labeling_scheme, "ernie"))
    
    data = torch.tensor(data, device=device, dtype=torch.float32).unsqueeze(0)
    print(f"data shape: {data.shape}")
    print(f"data dtype: {data.dtype}")
    print(f"data device: {data.device}")
    mask = data.clone().to(dtype=torch.int64, device=device)
 
    
    # synth = brainsynth.Synthesizer(
    #     brainsynth.config.SynthesizerConfig(
    #         builder="OnlySynth",
    #         out_size=out_size,
    #         out_center_str=out_center_str,
    #         segmentation_labels="ernie",
    #         device=device,
    #     )
    # )
    
    # #convert the data to a tensor
    
    # #check if im using mps
    # if device.type == "mps":
    #     #convert to float32
    #     data = data.astype(np.float32)
    
    # # After creating your images dictionary, add:
    # #mask = torch.zeros_like(data, dtype=torch.int64, device=device)
    
    # images = {"t1w": data, "generation_labels_dist": mask}
    
    # result = synth(images, unpack=False)
    
    # print(f"result keys: {result.keys()}")
    
    
    # #save the result
    # toSave = nib.Nifti1Image(result["image"][0].cpu().numpy(), affine, img.header)
    # nib.save(toSave, "result.nii.gz")
    
    # #save the t1w image
    # toSave = nib.Nifti1Image(result["t1w"][0].cpu().numpy(), affine, img.header)
    # nib.save(toSave, "t1w.nii.gz")

    # print("Result saved for default synthesizer")
    
    # Now let's try the Sav synthesizer
    
    synth = brainsynth.Synthesizer(
        brainsynth.config.SynthesizerConfig(
            builder="SaverioSynth",
            out_size=out_size,
            out_center_str=out_center_str,
            segmentation_labels="ernie",
            device=device,
        )
    )
    
    image_dict = dict(segmentation=mask)
    
    result = synth(image_dict, unpack=False)
    save_nifti_images(img, result, "0")
    
    for k in range(5):
        result = synth(image_dict, unpack=False)
        save_nifti_images(img, result, str(k))
        
    
