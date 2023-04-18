import cv2
import os
import numpy as np
import cv2
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














fix_h = 270
fix_w = 190



class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        for label in range(10):
            folder_path = f"{data_dir}/{label}/"
            for image_path in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, image_path))
                self.labels.append(label)
                # print(os.path.join(folder_path, image_path), label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
        return torch.from_numpy(image), torch.tensor(label)


def listdir(path, list_name, list_name2, type = '.csv'):  # , list_name2):
    # temp = os.listdir(path)
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name, list_name2,type=type)  # list_name2)
        elif os.path.splitext(file_path)[1] == type:
            list_name.append(file_path)
            list_name2.append(file)

def check_folder(log_folder, FLAG=1):
    if not os.path.exists(log_folder) and FLAG:
        os.makedirs(log_folder)
        print(f"{log_folder} is OK")

def addG(src,v = 100):
    # Add Gaussian noise
    mean = 0
    variance = v
    sigma = np.sqrt(variance)
    gaussian_noise = np.random.normal(mean, sigma, src.shape)
    noisy_img = src + gaussian_noise*10
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)  # clip values between 0 and 255
    return noisy_img

# def addP(src):
#     s_vs_p = 0.5
#     amount = 0.05
#     salt_pepper_noise = np.zeros(src.shape, dtype=np.uint8)
#     num_salt = np.ceil(amount * src.size * s_vs_p)
#     num_pepper = np.ceil(amount * src.size * (1.0 - s_vs_p))
#     coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in src.shape]
#     coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in src.shape]
#     salt_pepper_noise[coords_salt] = 255
#     salt_pepper_noise[coords_pepper] = 0
#     cv2.imshow("salt_pepper_noise",salt_pepper_noise)
#     cv2.waitKey(0)
#     noisy_img = cv2.add(src, salt_pepper_noise)
#     return noisy_img

def load_temp(path):
    temp = []

    K = 3
    for i in range(10):
        temp_frame = cv2.imread(f"{path}/{i}.png")
        print(temp_frame.shape)

        temp_frame = cv2.resize(temp_frame,(int(fix_w/K),int(fix_h/K)))
        temp_frame = cv2.medianBlur(temp_frame, 5)  # 奇数
        bin = cv2.inRange(temp_frame,np.array([0,0,120]),np.array([100,100,255]))
        cv2.imwrite(f"./data/{i}/{i}-o.jpg",bin)

        # Rotation
        rows, cols = bin.shape[:2]
        check_folder(f"./data/{i}/")
        # 54X27
        for angle in range(-6, 6, 2):
            for dx in range(-4, 4, 2):
                for dy in range(-4, 4, 2):
                    for scalex in [2, 3, 5]:
                        for scaley in [2, 3, 5]:
                            for v in [10, 50, 120]:
                                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                                dst = cv2.warpAffine(bin, M, (cols, rows))
                                # dst = addG(dst, v)

                                M = np.float32([[1, 0, dx], [0, 1, dy]])
                                dst = cv2.warpAffine(dst, M, (cols, rows))
                                # dst = addG(dst, v)

                                # Erosion
                                kernel = np.ones((scalex, scaley), np.uint8)
                                dst = cv2.erode(dst, kernel, iterations=1)
                                dst = addG(dst, v)

                                cv2.imwrite(f"./data/{i}/{i}-rotated_erode{angle}X{scalex}X{scaley}X{dx}X{dy}X{v}.jpg", dst)

        for angle in range(-6, 6, 2):
            for dx in range(-4, 4, 2):
                for dy in range(-4, 4, 2):
                    for scalex in [2, 3, 5]:
                        for scaley in [2, 3, 5]:
                            for v in [10, 50, 120]:
                                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                                dst = cv2.warpAffine(bin, M, (cols, rows))
                                # dst = addG(dst, v)

                                M = np.float32([[1, 0, dx], [0, 1, dy]])
                                dst = cv2.warpAffine(dst, M, (cols, rows))
                                # dst = addG(dst, v)

                                # dilate
                                kernel = np.ones((scalex, scaley), np.uint8)
                                dst = cv2.dilate(dst, kernel, iterations=1)
                                dst = addG(dst, v)

                                cv2.imwrite(f"./data/{i}/{i}-rotated_dilate{angle}X{scalex}X{scaley}X{dx}X{dy}X{v}.jpg", dst)

        # cv2.imshow(f"{i}bin", bin)
        # cv2.waitKey(0)
        # temp.append(bin)
    return temp

def check_num(list_num,roi):
    vlue = []
    h,w = roi.shape
    K_H = fix_h/h
    K_W = fix_w/w
    roi = cv2.resize(roi, (int(w*K_W), int(h*K_H)))
    for i in range(list_num.__len__()):
        # print(list_num[i].shape)
        # print(roi.shape)
        tem = cv2.absdiff(list_num[i], roi)
        # print(tem)
        vlue.append(np.sum(np.sum(tem,axis=0),axis=0))
        cv2.imshow(f"{i}", tem)

    # print(vlue)
    return np.argmin(np.array(vlue))

def frame2ROI(frame):

    pass


def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.to(torch.float32)
        target = target.to(torch.int64)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = criterion(output, target)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_dataloader.dataset)
    accuracy = 100. * correct / len(train_dataloader.dataset)
    print("train_loss,accuracy",train_loss,accuracy)
    return train_loss, accuracy

def validate(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            data = data.to(torch.float32)
            target = target.to(torch.int64)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            pred_np = pred.cpu().numpy()
            tru = target.cpu().numpy()
            indx = np.where((tru == 3) | (tru == 9))
            print("pred:", pred_np[indx].T)
            print("True:", tru[indx].T)
            print()
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_dataloader.dataset)
    accuracy = 100. * correct / len(val_dataloader.dataset)
    print("val_loss,accuracy",val_loss,accuracy)

    return val_loss, accuracy
if __name__ == '__main__':
    # load_temp("./material")
    #
    # exit(0)
    data_dir = "./data"
    batch_size = 32

    dataset = ImageDataset(data_dir)

    train_size = int(len(dataset) * 0.6)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my.MyCNN().to(device)
    # model = my.MyCNN.
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 50):
        train(model,train_dataloader,optimizer,criterion,device)
        validate(model,train_dataloader,criterion,device)
        print("epoch:", epoch)
    torch.save(model.state_dict(), 'my_model_new.pth')
    # Train your model using the dataloader
    # for batch in train_dataloader:
    #     images, labels = batch
    #     print(images.shape,labels)
        # Do your training steps here
