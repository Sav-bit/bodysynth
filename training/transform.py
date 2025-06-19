import os
import torch
import torchio as tio

class RandomDenormalization(tio.Transform):
    """
    A custom TorchIO transform to simulate non-normalized intensities.
    Instead of normalizing, this transform randomly scales and shifts
    the intensities of each image.
    
    For example, if your input image is normalized to [0, 1],
    applying a scale factor and an offset will produce a wider intensity range.
    Adjust the ranges as needed to mimic your target data.
    """
    def __init__(self, scale_range=(0.5, 1.5), offset_range=(-0.5, 0.5), p=1):
        super().__init__(p=p)
        self.scale_range = scale_range
        self.offset_range = offset_range

    def apply_transform(self, subject):
        scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
        offset = torch.FloatTensor(1).uniform_(*self.offset_range).item()
        # Apply scaling and offset to all scalar images in the subject
        for image in subject.get_images(intensity_only=True):
            image[tio.DATA] = image[tio.DATA] * scale + offset
        return subject

def load_subject(t1_path, seg_path):
    """
    Load the image and segmentation from file paths and create a TorchIO Subject.
    """
    t1 = tio.ScalarImage(t1_path)
    seg = tio.LabelMap(seg_path)
    subject = tio.Subject(t1=t1, seg=seg)
    return subject

def create_transforms():
    """
    Create a composed transform pipeline that applies:
      - Elastic deformations to simulate anatomical variability,
      - Affine transformations for rotation/translation/scaling,
      - Anisotropy to simulate resolution differences,
      - Gamma and bias field adjustments to modify intensities,
      - Random noise and spike artifacts to mimic lower quality scans.
    """
    elastic_transform = tio.RandomElasticDeformation(
        num_control_points=7,   # fewer control points for smoother, more global deformations
        max_displacement=15,    # stronger deformations for anatomical variability
        image_interpolation='bspline',
        locked_borders=2,
        p=1.0
    )
    
    affine_transform = tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=10,
        translation=5
    )
    
    anisotropy_transform = tio.RandomAnisotropy(
        p=0.5,
        downsampling=3,
        image_interpolation='bspline'
    )
    
    gamma_transform = tio.RandomGamma(
        p=0.7,
    )
    
    bias_transform = tio.RandomBiasField(
        p=0.5,
        coefficients=0.5,
        order=3,
    )
    
    noise_transform = tio.RandomNoise(
        mean=0,
        std=(0, 0.5),  # Adjust standard deviation to control noise level
        p=0.5
    )
    
    blur_transform = tio.RandomBlur(
        p=0.5,
        std=(0, 2),  # Adjust standard deviation to control blur level
    )
    
        # Custom transform to simulate non-normalized intensities
    denormalization_transform = RandomDenormalization(
        scale_range=(0.5, 1.5),     # Randomly scale intensities
        offset_range=(-0.5, 0.5),   # Randomly shift intensities
        p=1
    )
    
    # Compose all transforms into one pipeline
    transform = tio.Compose([
        elastic_transform,
        affine_transform,
        anisotropy_transform,
        gamma_transform,
        bias_transform,
        noise_transform,
        blur_transform,
        denormalization_transform,
    ])
    return transform

def generate_synthetic_patient(subject, transform, output_dir, index):
    """
    Apply the composed transforms to generate a synthetic patient and save the outputs.
    """
    transformed_subject = transform(subject)
    
    os.makedirs(output_dir, exist_ok=True)
    save_to = lambda file: os.path.join(output_dir, file)
    
    transformed_subject.t1.save(save_to(f'{index}_synthetic_T1.nii.gz'))
    transformed_subject.seg.save(save_to(f'{index}_synthetic_seg.nii.gz'))
    print(f"Synthetic patient {index} saved.")
    
def getMaxIndexOfExistingFile(directory):
    """
    Get the maximum index of existing files in the directory.
    """
    max_index = -1
    for filename in os.listdir(directory):
        if filename.endswith(".nii.gz"):
            try:
                index = int(filename.split('_')[0])
                max_index = max(max_index, index)
            except ValueError:
                continue
    return max_index
    

def main():
    # Define input paths for your single patient
    t1_path = "ErnieExtended/m2m_ernie_extended/T1.nii.gz"
    seg_path = "ernie_less_dim.nii.gz"
    
    # Load the subject
    subject = load_subject(t1_path, seg_path)
    
    # Create the transformation pipeline
    transform = create_transforms()
    
    # Output directory where synthetic data will be saved
    output_dir = "synt-2"
    
    
    # Get the maximum index of existing files in the output directory
    max_index = getMaxIndexOfExistingFile(output_dir)
    print(f"Max index of existing files: {max_index}")
    # If no existing files, start from 0
    if max_index == -1:
        max_index = 0
    else:
        max_index += 1
        
    print(f"Starting from index: {max_index}")
    
    # Generate multiple synthetic patients
    num_synthetic_patients = 2  # Adjust number as needed
    for idx  in range(num_synthetic_patients):
        generate_synthetic_patient(subject, transform, output_dir, idx + max_index)

if __name__ == "__main__":
    main()
