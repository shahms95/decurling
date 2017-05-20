# decurling

gen_data.py : script to synthetically generate curled images from normal images and cage file for training purposes // need to have Thea installed

script_distort_undistort.sh : contains commands to generate curled image from normal image and cage file as input as well as generating original image from distorted image and predicted cage file.

data directory : contains some images and the corresponding labels (cage files) to get started with training the neural network. Use the gen_data.py script to generate more such data.

NN directory : contains code for Neural Network

GenData directory : contains cage files to generate distorted images

pdfs_for_datasets : contains a few sample PDF files for generating data and a script to convert pdf to images
