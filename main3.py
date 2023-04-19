import cv2
import Tool as tool
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import torch.nn.functional as F
import MYCNN.model_CNN as my
import csv

TOPIC = "Ini"
# INPUT_FOLDER = "./video/23.4.12/"
OUTPUT_FOLDER = "./csv_file/"
tool.check_folder(OUTPUT_FOLDER)
# m
img = None
Gray = None
Bin = None
Draw = None
Temp_img1 = None
Temp_img2 = None

File_name = []
Path = []

# tool.listdir(INPUT_FOLDER, list_name=File_name, list_name2=Path, type=".mov")



def framROI(frame_100):
    gray_100 = cv2.cvtColor(frame_100, cv2.COLOR_BGR2GRAY)
    _, bin_100 = cv2.threshold(gray_100, 140, 255, cv2.THRESH_BINARY)
    bin_100 = 255 - bin_100
    bin_100 = cv2.resize(bin_100, (63, 90))
    data_to_process = cv2.merge([bin_100, bin_100, bin_100])
    data_to_process = cv2.cvtColor(data_to_process, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    data_to_process = data_to_process.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    data_to_process = np.transpose(data_to_process, (2, 0, 1))  # Convert HWC to CHW
    return torch.from_numpy(data_to_process[None,:,:,:]),bin_100

for VideoIndex in range(1):
    end_flag = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my.MyCNN().to(device)


    path1_model_path = './my_model_new.pth'
    path2_video_path = "./video/23.4.12/2.mov"
    path3_videosave_path = './save_demo.avi'
    path4_csvsave_path = "./2.mov-fast.csv"

    path5_num100_path = './2-100.npy'
    path6_num10_path = './2-10.npy'
    path7_num1_path = './2-1.npy'







    
    model.load_state_dict(torch.load(path1_model_path))
    cap = cv2.VideoCapture(path2_video_path)
    out_origin_ = cv2.VideoWriter(path3_videosave_path,cv2.VideoWriter_fourcc(*'XVID'), 30.0,
                                       (int(cap.get(4)), int(cap.get(3))), True)
    csvfile = open(path4_csvsave_path, "w", encoding="UTF-8")
    lomy_list = np.load(path5_num100_path)
    lomy_list2 = np.load(path6_num10_path)
    lomy_list3 = np.load(path7_num1_path)
    print(lomy_list)
    print(lomy_list)
    print(lomy_list)

    writer = csv.writer(csvfile, lineterminator='\n')
    writer.writerow(["frmae_number", "PH"])
    frame_num = 0
    while end_flag == 1:
        _, frame = cap.read()

        if frame.shape[0]==0:
            break

        frame_num = frame_num+1
        if frame_num % 30 ==0:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
            frame = cv2.medianBlur(frame, 3)  # 奇数

            frame_100 = frame[lomy_list[0][1]:lomy_list[1][1],
                    lomy_list[0][0]:lomy_list[1][0], :]

            frame_10 = frame[lomy_list2[0][1]:lomy_list2[1][1],
                    lomy_list2[0][0]:lomy_list2[1][0], :]

            frame_1 = frame[lomy_list3[0][1]:lomy_list3[1][1],
                    lomy_list3[0][0]:lomy_list3[1][0], :]
            # bin = cv2.inRange(frame,np.array([0,0,0]),np.array([170,170,170]))

            data_to_process_100,bin_100 = framROI(frame_100)
            data_to_process_10,bin_10 = framROI(frame_10)
            data_to_process_1,bin_1 = framROI(frame_1)

            data_to_process = torch.cat([data_to_process_100, data_to_process_10, data_to_process_1], dim=0)

            data = data_to_process.to(torch.float32).to(device)
            model.eval()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).cpu().numpy()
            result = pred[0,0]+pred[1,0]/10+pred[2,0]/100


            print("result:", result)
            cv2.putText(frame, f"{pred[0,0]}", (lomy_list[0][0], lomy_list[0][1]), 2, 1, (255,255,0), 2)
            cv2.putText(frame, f"{pred[1,0]}", (lomy_list2[0][0], lomy_list2[0][1]), 2, 1, (255,255,0), 2)
            cv2.putText(frame, f"{pred[2,0]}", (lomy_list3[0][0], lomy_list3[0][1]), 2, 1, (255,255,0), 2)
            cv2.putText(frame, f"{int(frame_num/30)}", (30,30), 2, 1, (255,0,0), 2)
            writer.writerows([[frame_num, result]])
            out_origin_.write(frame)
            cv2.imshow(f"{VideoIndex}bin", frame)
            cv2.imshow(f"{VideoIndex}bin_100", bin_100)
            cv2.imshow(f"{VideoIndex}bin_10", bin_10)
            cv2.imshow(f"{VideoIndex}bin_1", bin_1)
            key = cv2.waitKey(1)
            if key == 27:
                break
    csvfile.close()


