import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: Cole-Anticevic R and L subcortical parcellation
pathtomask1 =r'\\wsl.localhost\Ubuntu-20.04\...\cole_anticevic\cole_anticevic_subcortex_atlas_GSR_R.nii'
gl.overlayload(pathtomask1)
pathtomask2 =r'\\wsl.localhost\Ubuntu-20.04\...\cole_anticevic\cole_anticevic_subcortex_atlas_GSR_L.nii'
gl.overlayload(pathtomask2)
gl.opacity(0,30)
gl.opacity(1,100)
gl.opacity(2,100)
gl.colorname(1,"x_rain")
gl.colorname(2,"x_rain")
gl.colorbarposition(0)
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("S R 0; S 0");
# save image as png
gl.savebmp(os.path.join(dir, 'ColeAnticevicSubcortexL+R.png'))
