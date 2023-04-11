# Jetson Environment
Create environment with following steps

# Installing Wheels for Torch and Torchvision
Simply go to Your browsers and paste these links to get whl files for torch and torchvision
https://drive.google.com/uc?id=1aWuKu8eqkZwVzFFvguVuwkj0zdCir9qX
https://drive.google.com/uc?id=1P0xyPT-WIWglqmT195OSyazV_1LPaHDa
go to the folder where these are downloaded.
Open that directory in terminal and run following commands.

# Install all the dependencies
sudo chmod 777 env_setup.sh
./env_setup.sh

# Run App
The above process will take a while, after that you can run GUI.

# Training
Save the images in the same name of folder of the person in face_database folder.
A sample database has been provided.

# GUI
Double click the GUI Icon to execute the application.
It has 3 buttons. First is to train then image inference and then tracking.

Make sure you have faces_database.pickle file in models before you start inference.
If not you can first train on sample database of the database of yours. Keep the heirarchy as of sample database.
Ideally: Click the train button make sure camera is conected first we enter the ID and Name  required person is entered correctly. An inference window will pop up on right side of the GUI as the process begins.

To run image inference. It is required to give an image to test if detections are working correctly.

To ruk tracker, Make sure you have a usb camera attached to your jetson. An inference window will pop up on right side of the GUI as the process begins.

# Detection:
The code is designed to detect if the person detected is wearing a mask or not. If mask is detected the GUI will send a request to person to remove the mask so that it can start detecting the person correctly to mark the attendance. Please make sure to have good lightened up environment for good accuracy in all the aspects whether its training, inference or mask detection.

# Data Collection
At every run, data will be saved in the csv files. Every run will make a new csv file in the folder as you run it. If the name of the person detected matches to the name entered in the bar, The application will write a row in the csv for confirmation with timestamp.




# Send csv file using email
In line 47 in gui.py write the subject of email.
in line 48 the body of email where you want send any type of materials
in line 49 write your sender email. 
Note: follow the instruction of video to how can you allow the less secure app in your account.. 
in line 50 add the reciver email..
in line 50 write the sender email account password

