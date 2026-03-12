import gl
import os
# directory for saving images
dir = r'\\wsl.localhost\Ubuntu-20.04\...\MRIcroGL_visuals'
gl.resetdefaults()
gl.backcolor(255, 255, 255)
#open background image
gl.loadimage('mni152')
#open overlay: Yeo7
pathtomask =r'\\wsl.localhost\Ubuntu-20.04\...\nilearn_data\yeo_2011\Yeo_JNeurophysiol11_MNI152\Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz'
gl.overlayload(pathtomask)
gl.opacity(0,30)
gl.opacity(1,100)
gl.colorname(1,"x_rain")
gl.colorbarposition(0)
gl.shaderadjust("brighten", 100)
#"a"xial, "c"oronal and "s"agittal "r"enderings
gl.mosaic("S R 0; S 0");
# save image as png
gl.savebmp(os.path.join(dir, 'Yeo7.png'))
