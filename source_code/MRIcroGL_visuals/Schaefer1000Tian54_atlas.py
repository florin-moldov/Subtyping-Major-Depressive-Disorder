import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: Integrated Schaefer 1000 + Tian 54
pathtomask =r'\\wsl.localhost\Ubuntu-20.04\...\Schaefer1000_TianS4_combined.nii.gz'
gl.overlayload(pathtomask)
gl.opacity(0,30)
gl.opacity(1,100)
gl.colorbarposition(0)
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("S R 0; S 0");
# save image as png
gl.savebmp(os.path.join(dir, 'Schaefer1000Tian54_atlas.png'))
