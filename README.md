# Multimodal technology for age and metallicity estimation
The provided folder contains example data and related code files, with SDSS image data and corresponding spectral data stored in separate subdirectories. The file descriptions are as follows:
1. data00_download_images.py: Code for image download;
2. data01_fit_pre.py: Code for spectral data preprocessing;
3. data02image_pre.py: Code for image data preprocessing;
4. data03_read_to_npz.py: Read image and spectrum files as npz file;
5. model01m1_1_fits_pre.py, model03m2_1_images_to_fits.py, model04m3_1_images_pre.py, model06m4_mul_pre.py: Implementations of Spectral Feature Extraction Model (M1), Simulated Spectral Feature Generation Model (M2), Image Feature Extraction Model (M3), and Multimodal Attention Regression Model (M4), respectively;
6. model02m1_2_real_fit_feature.py: Code for feature extraction of all spectral data using model M1;
7. run_scripts.sh: Script to automatically run all code in Linux system
8. The provided CSV file includes image filenames, corresponding spectral filenames, and associated galaxy parameters.

The runtime environment is Python 3.8.5 and TensorFlow-GPU 2.10.0.
Additionally, the multimodal attention model (M4) can be used to fuse features from image and spectral data, enabling predictions of galaxy age and metallicity. Due to the large size of the model file, if you need the file or have further questions, please contact the author via email at liping990523@163.com.
