
Chess Sample - v1 2021-07-06 1:13pm
==============================

This dataset was exported via roboflow.ai on July 6, 2021 at 5:15 AM GMT

It includes 30 images.
Pieces are annotated in YOLO v4 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 30 percent of the image
* Random rotation of between -5 and +5 degrees
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -25 and +25 percent
* Salt and pepper noise was applied to 2 percent of pixels


