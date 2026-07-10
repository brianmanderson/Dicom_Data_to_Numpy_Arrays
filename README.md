# Dicom_Data_to_Numpy_Arrays

> **Deprecated.** This code is superseded by
> [Dicom_RT_and_Images_to_Mask](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask),
> which does the same conversion (also in parallel). Kept here for reference; last updated 2020.

Parallel conversion of DICOM image series and their RT structure sets into numpy
image/mask arrays for training deep-learning segmentation models. Given a tree of
patient DICOM folders and a list of contour names (with an "associations" dictionary
mapping ROI-name variants to canonical names), it finds each patient's RTSTRUCT file,
checks that the requested contours exist, rasterizes the contours into masks, and
writes the paired image/mask arrays.

## Components

- `DicomImagesintoData_Parallel.py` — main pipeline. Multi-threaded workers (one queue
  to verify RT structures, one to convert) walk patient directories and build the
  arrays using pydicom, SimpleITK, and skimage polygon drawing.
- `Separate_Numpy_Images_Into_Test_Train_Validation.py` — splits the generated
  image/mask files into Train / Test / Validation folders by patient.
- `Make_Patient_pickle_file_from_text.py` — builds the patient-info pickle consumed by
  the pipeline.
- `Get_Path_Info.py` — indexes processed folders into a `Data_Locations.pkl`.
- `Utils.py` — pickle load/save helpers and a matplotlib scroll-wheel slice viewer
  (`plot_scroll_Image`) for eyeballing the resulting volumes.

## Requirements

Python with pydicom, SimpleITK, numpy, scikit-image, and matplotlib.
