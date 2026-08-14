import os
import shutil
import subprocess
import sys

try:
	import nibabel as nib
except Exception:
	sys.exit('请先安装 nibabel: pip install nibabel')


def convert_mgz_to_nii(mgz_path, nii_path):
	if not os.path.exists(mgz_path):
		raise FileNotFoundError(mgz_path)
	# 首先尝试使用 nibabel 直接读取并保存为 nii.gz
	try:
		img = nib.load(mgz_path)
		nib.save(img, nii_path)
		return
	except Exception:
		# 如果 nibabel 失败且系统上有 FreeSurfer 的 mri_convert，则使用它
		if shutil.which('mri_convert'):
			subprocess.check_call(['mri_convert', mgz_path, nii_path])
			return
		raise


if __name__ == '__main__':
	base = r'D:\python_code\projects\thesis\outputs\cortex_seg\patient001\mri'
	# convert_mgz_to_nii(os.path.join(base, 'aparc.DKTatlas+aseg.deep.mgz'), os.path.join(base, 'aparc.DKTatlas+aseg.deep.nii.gz'))
	# convert_mgz_to_nii(os.path.join(base, 'mask.mgz'), os.path.join(base, 'mask.nii.gz'))
	convert_mgz_to_nii(os.path.join(base, 'orig.mgz'), os.path.join(base, 'orig.nii.gz'))