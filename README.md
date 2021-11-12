# Object Segmentation By Point Tracking
This repo includes implementation of object segmentation based on point tracking accross multiple frames.

## Specification

All merge and working code are located in source_code folder

## Instruction

Run main.py will generate result of 2 frames 22 and 23 based in /src/images/Marple13_eig

## Usage

_ threshold.py: return threshold of the image.

_ occlusion.py: perform occlusion detection and return which point allowed to track.

_ affinity.py: contains code for building affinity matrix.

_ track.py: object Track to keep history of trajectory.

_ tracking.py: contains code for trajectory tracking.

_ spectral_clustering: returns point with label after cluster.

_ main.py: call all the functions and display result.

_ Other python files have smaller function that support other calculation as well as debugging.

## Documents

_ A summary of overall of our project: Automatic_Object_Segmentation_Poster.pdf
_ All our findings and documetations is located in: Automatic_Object_Segmentation_Report.pdf

## Contribution

This project is work of:\
Duc Tran (dt2259@nyu.edu)\
Andrew Weng (aw4108@nyu.edu)\
Russell Wustenberg (rw2873@nyu.edu)\
Ye Xu (yx2534@nyu.ed)

## License
[MIT](https://choosealicense.com/licenses/mit/)
