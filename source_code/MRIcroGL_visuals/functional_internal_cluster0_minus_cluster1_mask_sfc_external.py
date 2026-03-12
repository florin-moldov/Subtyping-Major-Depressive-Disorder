import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: difference mask cluster 0 > cluster 1 (internal functional connectivity for sfc external clustered MDD subjects)
pathtomask =r'\\wsl.localhost\Ubuntu-20.04\...\nifti_masks\F32_functional_internal_cluster0_minus_cluster1_mask_sfc_external.nii.gz'
gl.overlayload(pathtomask)
gl.opacity(0,40)
gl.opacity(1,100)
gl.colorname(1,"HOTIRON")
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("A R 0 S R 0; A 0 S 0");
# save image as png
gl.savebmp(os.path.join(dir, 'functional_internal_cluster0_minus_cluster1_sfc_external.png'))
