# Object Detection Using Keras RetinaNet 

## Installation

1) Ensure numpy is installed using `pip install numpy --user`
2) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
3) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.

## Doenloading weights

To download the weights run `weight_download.sh`
You can also download weights manually from "https://github.com/fizyr/keras-retinanet/releases" named "resnet50_coco_best_v2.1.0.h5" and place it in the "snapshots" directory.



## Testing

For testing run "ObjectDetector.py" with --img_dir as path to input image directory and --det as path to output result directory.

