# Necessary Imports
import cv2
import tkinter as tk
from tkinter import Canvas, NW, StringVar
from tkinter.ttk import Label
from tkinter.font import Font
import torch
import numpy as np
import os
from PIL import Image, ImageTk
import csv
from csv import writer
from models.utils_facenet import mask_detect
from datetime import datetime
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



# Load Mask Detection Model
device = torch.device("cpu")
# device = torch.device("cuda:0")
with open('models/svm.pickle', 'rb') as p1:
    svm_model = pickle.load(p1)

# Load Face detection model. Will download itself if doesn't exist.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=60,
    thresholds=[0.3, 0.3, 0.3], factor=0.709, post_process=True,
    device=device
)

# Basic parameters
workers = 0 if os.name == 'nt' else 4
face_detection_reset_time = 20
verification_constant = 10
recognition_threshold = 0.85

subject = "An email with csv file" # write subject of email
body = "This is an email with attachment of cv file"  # write body  message of email
sender_email = "muzammilf365@gmail.com"  # sender email and allow the less secure app video also include in the project how to allow the less secure app of sender email
receiver_email = "muzammil@projectz.io" #  receiver email name
password = 'Allah786!@#' # sender email password
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject
message["Bcc"] = receiver_email


# Writing files to csv function
def append_list_as_row(file_name, list_of_elem, write_header):
    if write_header == True:
        fieldnames = ['Time Stamp', 'Student Name', 'Student ID']
        with open(file_name, 'a+', newline='') as csvfile:
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerow(
                {fieldnames[0]: list_of_elem[0], fieldnames[1]: list_of_elem[1], fieldnames[2]: list_of_elem[2]})
    else:
        with open(file_name, 'a+', newline='') as write_obj:

            csv_writer = writer(write_obj)

            csv_writer.writerow(list_of_elem)

# Main GUI Class
class App:

    def __init__(self, window, window_title, file_name):
        self.writer = file_name
        self.window = window
        self.window.title(window_title)
        self.width = self.window.winfo_screenwidth()

        self.height = self.window.winfo_screenheight()
        self.window.geometry("%dx%d" % (self.width, self.height))
        self.write_header = True
        self.training_message = StringVar()
        self.training_message.set('')

        self.track_message = StringVar()
        self.track_message.set('No Video Loaded')

        self.id_message = StringVar()
        self.id_message.set('')

        self.name_message = StringVar()
        self.name_message.set('')

        self.canvas = Canvas(window, width=int(self.width * 0.343), height=int(self.height * 0.556))
        self.canvas.place(x=int(self.width * 0.5936), y=int(self.height * 0.2222))

        self.canvas_logo = Canvas(self.window, width=int(self.width * 0.172), height=int(self.height * 0.278))
        self.canvas_logo.place(x=int(self.width * 0.0139), y=int(self.height * 0.0111))
        logo_picture = cv2.imread('logo.png')
        logo_picture = cv2.resize(logo_picture, (int(self.width * 0.172), int(self.height * 0.278)))
        self.photo_logo = ImageTk.PhotoImage(image=Image.fromarray(logo_picture))
        self.canvas_logo.create_image(0, 0, image=self.photo_logo, anchor=NW)

        frame_resized = np.zeros((550, 500))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.create_widgets()

        self.window.mainloop()

    # GUI Widgets
    def create_widgets(self):
        myfont = Font(family="Times", size=20, weight="bold", underline=1)
        label_0 = Label(self.window, text="Face-Recognition-Based-Attendance-Management-System", width=100, font=myfont)
        label_0.place(x=self.width // 4, y=self.height // 11)
        Callisto = Font(family="Times", size=12, weight="bold", underline=1)

        label_1 = Label(self.window, text="Enter Id", width=20, font=("bold", 10))
        label_1.place(x=int(self.width * 0.3125), y=int(self.height * 0.2556))
        self.inputtxt_1 = tk.Text(
            height=1,
            width=20)
        self.inputtxt_1.place(x=int(self.width * 0.3875), y=int(self.height * 0.2556))
        label_2 = Label(self.window, text="Enter Name", width=20, font=("bold", 10))
        label_2.place(x=int(self.width * 0.3125), y=int(self.height * 0.3222))
        self.inputtxt_2 = tk.Text(
            height=1,
            width=20)
        self.inputtxt_2.place(x=int(self.width * 0.3875), y=int(self.height * 0.3222))

        label_3 = Label(self.window, text="Notification:", width=20, font=Callisto)
        label_3.place(x=int(self.width * 0.3125), y=int(self.height * 0.4222))

        btn2 = tk.Button(self.window, text='Train Images ', width=10, height=3,
                         command=self.train_images)
        btn2.place(x=self.width // 8, y=self.height // 2)

        btn3 = tk.Button(self.window, text='Track Images ', width=10, height=3,
                         command=self.track)
        btn3.place(x=int(self.width * 0.3125), y=self.height // 2)
        self.label_8 = Label(self.window, textvariable=self.track_message, width=20, font=("bold", 10))
        self.label_8.place(x=int(self.width * 0.3125), y=int(self.height * 0.57222))

        self.label_9 = Label(self.window, textvariable=self.id_message, width=20, font=("bold", 8))
        self.label_9.place(x=int(self.width * 0.39375), y=int(self.height * 0.28889))

        self.label_10 = Label(self.window, textvariable=self.name_message, width=20, font=("bold", 8))
        self.label_10.place(x=int(self.width * 0.39375), y=int(self.height * 0.35556))

        btn4 = tk.Button(self.window, text='Quit ', width=10, height=3,
                         command=self.window.destroy)
        btn4.place(x=int(self.width * 0.5), y=self.height // 2)

        self.label_7 = Label(self.window, textvariable=self.training_message, width=20, font=("bold", 10))
        self.label_7.place(x=int(self.width * 0.121875), y=int(self.height * 0.57222))

    # Training Function
    def train_images(self):

        # Checks if database exists, otherwise adds new faces to existing database
        previous_emb = []
        names = []
        new_embeddings = []
        try:
            with open('models/faces_database.pickle', 'rb') as p:
                database = pickle.load(p)
            previous_emb = database['embeddings']
            names = database['names']
            self.training_message.set('Database found')
            self.window.update_idletasks()
        except:
            self.training_message.set('Database not found')
            self.window.update_idletasks()
            pass

        frame_resized = np.zeros((550, 500))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.track_message.set('No Video Loaded')

        # Verifies if id and name has been entered in text box
        id = self.inputtxt_1.get("1.0", 'end-1c')
        name_enter = self.inputtxt_2.get("1.0", 'end-1c')
        if id == '':
            self.id_message.set('Please Enter ID')
            if name_enter != '':
                self.name_message.set('')
            self.window.update_idletasks()
        if name_enter == '':
            self.name_message.set('Student Name Missing')
            if id != '':
                self.id_message.set('')
            self.window.update_idletasks()

        # Runs training if ID and name added
        if id != '' and name_enter != '':
            # Captures live camera
            capture = cv2.VideoCapture(0)
            self.training_message.set('Capturing Images...')
            self.window.update_idletasks()
            i = 0
            frame_count = 0
            entered_cam = False

            # Waits until training of 15 frames per person
            while i < 15:
                ret, frame = capture.read()
                if ret:
                    entered_cam = True
                    img = frame.copy()
                    try:
                        img_pil = Image.fromarray(img)
                        all_boxes, _ = mtcnn.detect(img)

                        x_aligned, prob = mtcnn(img_pil, return_prob=True)
                        x_aligned_list = []

                        if x_aligned is not None:

                            box = [int(x) for x in all_boxes[0]]
                            x_aligned_list.append(x_aligned)

                            aligned = torch.stack(x_aligned_list).to(device)
                            # Generates encodings of faces one by one and saves inside of Tensors
                            embeddings = resnet(aligned).detach().cpu()

                            if new_embeddings == []:
                                new_embeddings = embeddings
                            else:
                                new_embeddings = torch.cat((new_embeddings, embeddings), 0)

                            names.append(f'{name_enter} {id}')

                            i += 1
                            perc = int(i / 15 * 100)
                            # Shows message at GUI about how much training is done
                            message = f'Training... {perc} %'
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), [255, 0, 0], 2)
                            cv2.putText(img, f'Training {name_enter}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        [0, 0, 255], 2, cv2.LINE_4)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                            self.training_message.set(message)
                            self.window.update_idletasks()
                    except:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                        self.window.update_idletasks()

                    frame_count += 1
                else:
                    break
            # Saves/Adds new encodings to database
            if entered_cam == True:
                message = f'{name_enter} Added to DB'
                self.training_message.set(message)
                self.window.update_idletasks()

                if previous_emb != []:
                    new_embeddings = torch.cat((previous_emb, new_embeddings), 0)

                database = {'embeddings': new_embeddings, 'names': names}

                with open('models/faces_database.pickle', 'wb') as p:
                    pickle.dump(database, p, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                message = f'No Cam Found'
                self.training_message.set(message)
                self.window.update_idletasks()

    # Tracker Function
    def track(self):

        self.name_message.set('')
        self.id_message.set('')
        self.window.update_idletasks()
        frame_resized = np.zeros((550, 500))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        # Checks if database is generated, otherwise asks to train faces first.
        try:
            with open('models/faces_database.pickle', 'rb') as p:
                database = pickle.load(p)
        except:
            database = {}
        if database != {}:
            database_embeddings = database['embeddings']
            database_names = database['names']
        else:
            self.training_message.set('Train Here...')
            self.window.update_idletasks()
            return

        video_path = 0
        # try:
        if True:
            # Captures the live camera
            vid = cv2.VideoCapture(video_path)
            enter = False
            count = 0
            flag_csv = 0
            self.track_message.set('Processing...')
            self.window.update_idletasks()
            student_count = 0
            students = []
            while True:
                ret, frame = vid.read()
                # try:
                if True:
                    if not ret:
                        if enter == True:
                            track_message = 'Tracker Exit'
                        else:
                            track_message = 'Tracker Load Error'
                        self.track_message.set(track_message)
                        self.window.update_idletasks()
                        frame_resized = np.zeros((550, 500))
                        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                        break
                    else:
                        if enter == False:
                            count += 1
                            if count >= 3:
                                enter = True
                        # Sends image to detect function to get name, mask on/off signal and image with bounding box
                        result_image, name_id, mask_lbl = mask_detect(frame, mtcnn, resnet, database_embeddings,
                                                                      database_names, recognition_threshold, device,
                                                                      svm_model)
                        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(rgb_image, (550, 500))
                        # Displays images to GUI
                        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                        self.window.update_idletasks()

                        # ID verification and no repeat ID checks
                        if name_id is not None and mask_lbl == name_id:
                            name = name_id.split(' ')[0]
                            id = name_id.split(' ')[1]
                            load_message = f'Verifying {name}'
                            self.track_message.set(load_message)
                            self.window.update_idletasks()
                            flag_csv += 1
                            if flag_csv >= verification_constant:

                                if id not in students:
                                    load_message = f'Attendance Marked'
                                    dt = str(datetime.now()).split('.')[0]
                                    students.append(id)
                                    # saves results in csv here
                                    row_contents = [dt, name, id]
                                    append_list_as_row(self.writer, row_contents, self.write_header)
                                    if self.write_header == True:
                                        self.write_header = False
                                    self.track_message.set(load_message)
                                    self.window.update_idletasks()
                                    student_count += 1
                                    flag_csv = 0


                                    message.attach(MIMEText(body, "plain"))

                                    if student_count >= 1:
                                        # Open PDF file in binary mode
                                        with open(self.writer, "rb") as attachment:
                                            # Add file as application/octet-stream
                                            # Email client can usually download this automatically as attachment
                                            part = MIMEBase("application", "octet-stream")
                                            part.set_payload(attachment.read())
                                        # Encode file in ASCII characters to send by email
                                        encoders.encode_base64(part)
                                        # Add header as key/value pair to attachment part
                                        part.add_header(
                                            "Content-Disposition",
                                            f"attachment; filename= {self.writer}",
                                        )

                                        # Add attachment to message and convert message to string
                                        message.attach(part)
                                        text = message.as_string()
                                        # Log in to server using secure context and send email
                                        context = ssl.create_default_context()
                                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                                            server.login(sender_email, password)
                                            server.sendmail(sender_email, receiver_email, text)
                                        break
                                else:
                                    self.track_message.set('Student ID Marked')
                                    self.window.update_idletasks()
                        elif mask_lbl != name_id and mask_lbl == 'Please Remove Mask':
                            load_message = f'Please Remove Mask'
                            self.track_message.set(load_message)
                            self.window.update_idletasks()
                            flag_csv = 0
                        else:
                            load_message = f'Not in Database'
                            self.track_message.set(load_message)
                            self.window.update_idletasks()
                            flag_csv = 0

                # except:
                #     pass

        # except:
        #     track_message = 'Tracker Load Error'
        #     self.track_message.set(track_message)
        #     self.window.update_idletasks()

# Main code runner
if __name__ == '__main__':
    try:
        os.mkdir('csv_files')
    except:
        pass
    j = 0
    out_file = f'csv_files/results_{j}.csv'
    while True:

        if os.path.exists(out_file):
            j += 1
            out_file = f'csv_files/results_{j}.csv'
        else:
            break

    # Shows at the beginning of the code where the data will be stored
    print('Data is saving in:', out_file)
    print('Process Begin...')

    App(tk.Tk(), 'Face Recognition GUI...', out_file)
