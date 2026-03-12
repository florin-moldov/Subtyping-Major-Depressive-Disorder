import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: module 7
pathtomask =r'\\wsl.localhost\Ubuntu-20.04\...\module_07_mask.nii.gz'
gl.overlayload(pathtomask)
gl.opacity(0,40)
gl.opacity(1,100)
gl.colorname(1,"7cool")
gl.colorbarposition(0)
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("A R 0 S R 0");
# save image as png
gl.savebmp(os.path.join(dir, 'module7.png'))
