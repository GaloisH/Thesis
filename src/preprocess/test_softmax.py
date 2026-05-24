import nibabel as nib

file_path=r'C:\Users\13765\Downloads\case_000_0000_softmax_prob.nii.gz'
if __name__ == "__main__":
    img = nib.load(file_path)
    shape=img.shape
    print(shape)
    for i in range(shape[-1]):
        nib.save(nib.Nifti1Image(img.dataobj[..., i], img.affine), rf"C:\Users\13765\Downloads\case_000_0000_softmax_prob_channel_{i}.nii.gz")