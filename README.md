# Multimodal technology for age and metallicity estimation
The provided folder contains example data and related code files, with SDSS image data and corresponding spectral data stored in separate subdirectories. The file descriptions are as follows:
1. 01fits_pre.py: Code for spectral data preprocessing;
2. 02image_pre.py: Code for image data preprocessing;
3. 03image_crop.py: Code for cropping image data;
4. 04_fits_pre.py, 05_images_pre.py, 06_mul_pre.py: Implementations of Spectral Feature Extraction Model (M1), Image Feature Extraction Model (M3), and Multimodal Attention Regression Model (M4), respectively;
5. 07_images_to_fits.py: Implementation of Simulated Spectral Feature Generation Model (M2);
The provided CSV file includes image filenames, corresponding spectral filenames, and associated galaxy parameters.
The runtime environment is Python 3.8.5 and TensorFlow-GPU 2.10.0.
Additionally, the multimodal attention model (M4) can be used to fuse features from image and spectral data, enabling predictions of galaxy age and metallicity. Due to the large size of the model file, if you need the file or have further questions, please contact the author via email at liping990523@163.com.
