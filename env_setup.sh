# Open the terminal in root directory and enter the following commands one by one to complete environment setup

sudo apt-get update 
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran 
sudo apt-get install python3-pip 
sudo pip3 install -U pip testresources
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo -H pip3 install future
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install Cython
sudo apt-get install python3-pil
sudo pip3 install imutils
sudo pip3 install opencv-python
sudo pip3 install thop
sudo pip3 install scikit-image
sudo pip3 install scikit-learn
sudo -H pip3 install gdown
sudo -H pip3 install torch-1.7.0a0-cp36-cp36m-linux_aarch64.whl
sudo -H pip3 install torchvision-0.8.0a0+291f7e2-cp36-cp36m-linux_aarch64.whl
sudo pip3 install facenet-pytorch
# After Environment is complete You can run GUI by Double Clicking the Icon.
# Refer to the guide provided on how to use the Application.
# Note: If gdown commands fail, You can manually go to the link the browser to download torch and torchvision files and proceed with next steps.
