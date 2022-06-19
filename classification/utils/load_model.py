import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
import pickle
from classification.models.MobileNetV2 import mobilenet_v2


class Model:
    def __init__(self, path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = str(path)
        self.load_model()

    def load_model(self):
        try:
            with open(self.path + '/model.pickle', 'rb') as file:
                self.model = pickle.load(file)
        except:
            # load custom model
            self.model = mobilenet_v2(pretrained=False, num_classes=2).to(self.device)
        with open(self.path + '/parameter.txt', 'r') as file:
            temp = file.readline()
            temp = temp.split('=')[1].strip('\n').strip()
            self.input_size = int(temp)
            file.close()
        self.model.load_state_dict(torch.load(self.path + "/best.h5", map_location=self.device))
        label = self.path + "/label.txt"
        file = open(label, 'r')
        str = file.readlines()
        file.close()
        self.class_name = []
        self.class_id = []
        for i in str:
            temp = i.split(':')
            self.class_name.append(temp[0].strip())
            self.class_id.append(int(temp[1].strip("\n").strip()))
        self.class_name_id = zip(self.class_name, self.class_id)
        self.class_name_id = dict(self.class_name_id)

    def preprocess_image(self, image):
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(image)
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        # img = Image.open(i).convert('RGB')
        input_size = self.input_size
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize])
        img_preprocessed = preprocess(img)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
        return batch_img_tensor

    def predict(self, image):
        img = self.preprocess_image(image)
        self.model.eval()
        out = self.model(img.to(self.device))
        _, index = torch.max(out, 1)
        percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
        #print(index)
        return self.class_name[index], percentage

