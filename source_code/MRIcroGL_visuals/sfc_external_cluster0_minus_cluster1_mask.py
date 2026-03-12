import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: difference mask cluster 0 > cluster 1 (external structure-function coupling)
pathtomask =r'\\wsl.localhost\Ubuntu-20.04\...\nifti_masks\F32_sfc_external_cluster0_minus_cluster1_mask.nii.gz'
gl.overlayload(pathtomask)
gl.opacity(0,40)
gl.opacity(1,100)
gl.colorname(1,"electric_blue")
# set value for 'brightest' to a value close to 0 but not 0 (otherwise it will color the background as well)
# max value here is max difference value in our mask
gl.minmax(1, -0.001, -0.04)
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("A R 0 S R 0; A 0 S 0");
# save image as png
gl.savebmp(os.path.join(dir, 'sfc_external_cluster0_minus_cluster1.png'))
