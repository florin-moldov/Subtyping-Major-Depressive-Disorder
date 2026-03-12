import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: module 9
pathtomask =r'\\wsl.localhost\Ubuntu-20.04\...\module_09_mask.nii.gz'
gl.overlayload(pathtomask)
gl.opacity(0,40)
gl.opacity(1,100)
# No brown color template, make it ourselves
gl.colornode(1,0,0,96,63,0,0)
gl.colornode(1,1,100,96,63,0,94)
gl.colornode(1,2,255,96,63,0,164)
gl.colorbarposition(0)
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("A R 0 S R 0");
# save image as png
gl.savebmp(os.path.join(dir, 'module9.png'))
