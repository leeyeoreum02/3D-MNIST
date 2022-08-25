from re import A
import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, id_list, label_list, point_list, mode):
        self.id_list = id_list
        self.label_list = label_list
        self.point_list = point_list
        self.mode = mode
        
    def __getitem__(self, index):
        image_id = self.id_list[index]
        
        if self.mode in ['train', 'val']:
            points = self.point_list[int(image_id)][:]
            # print(points.shape)
            points = self.randomRotate(points)
        elif self.mode == 'test':
            points = self.point_list[int(image_id)-50000][:]

        image = self.get_vector(points)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return torch.Tensor(image).unsqueeze(0), label
        else:
            return torch.Tensor(image).unsqueeze(0)
    
    def get_vector(self, points, x_y_z=[16, 16, 16]):
        # 3D Points -> [16,16,16]
        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
        xyzmin = xyzmin - diff / 2
        xyzmax = xyzmax + diff / 2

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num 
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)

        n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        n_x = x_y_z[0]
        n_y = x_y_z[1]
        n_z = x_y_z[2]

        structure = np.zeros((len(points), 4), dtype=int)
        structure[:,0] = np.searchsorted(segments[0], points[:,0]) - 1
        structure[:,1] = np.searchsorted(segments[1], points[:,1]) - 1
        structure[:,2] = np.searchsorted(segments[2], points[:,2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:,3] = ((structure[:,1] * n_x) + structure[:,0]) + (structure[:,2] * (n_x * n_y)) 

        vector = np.zeros(n_voxels)
        count = np.bincount(structure[:,3])
        vector[:len(count)] = count

        vector = vector.reshape(n_z, n_y, n_x)
        return vector
    
    def randomRotate(self, dots):
        a, b, c = np.random.randint(low=-32, high=33, size=3)
        if a != 0:
            a = np.pi/a
        if b != 0:
            b = np.pi/b
        if c != 0:
            c = np.pi/c
        mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        m = np.dot(np.dot(mx,my),mz)
        dots = np.dot(dots, m.T)
        return dots

    def __len__(self):
        return len(self.id_list)


class PointCloudDataset(Dataset):
    def __init__(self, id_list, label_list, point_list, npoints=1024, mode='train'):
        self.npoints = npoints
        self.id_list = id_list
        self.point_list = point_list
        self.label_list = label_list
        self.mode = mode
        self.device = torch.device('cuda:0')
        
    def __getitem__(self, index):
        image_id = self.id_list[index]

        if self.mode in ['train', 'val']:
            points = self.point_list[int(image_id)][:]
        elif self.mode == 'test':
            points = self.point_list[int(image_id)-50000][:]
        
        #randomly sample points
        choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        points = points[choice, :]
        
        #normalize to unit sphere
        points = points - np.expand_dims(np.mean(points, axis=0), 0) #center
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)), 0)
        points = points / dist #scale

        if self.mode == 'train':
            points = self.random_rotate(points, 0.2)
            points = self.random_jitter(points, 0.5)
        elif self.mode == 'val':
            points = self.random_rotate(points, 0.2)
            # points = self.random_jitter(points, 0.5)
        else:
            pass

        if self.label_list is not None:
            label = self.label_list[index]
            return torch.from_numpy(points).float(), torch.tensor(label)
        else:
            return torch.from_numpy(points).float()
        
    def __len__(self):
        return len(self.id_list)
    
    def random_rotate(self, points, threshold):
        # a, b, c = np.random.randint(low=-32, high=33, size=3)
        # if a != 0:
        #     a = np.pi/a
        # if b != 0:
        #     b = np.pi/b
        # if c != 0:
        #     c = np.pi/c
        prob = np.random.rand(1)
        if prob > threshold:
            a, b, c = np.random.uniform(-1*np.pi, np.pi, 3)
            mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
            mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
            m = np.dot(np.dot(mx,my),mz)
            points = np.dot(points, m.T)
        return points

    def random_jitter(self, points, threshold):
        prob = np.random.rand(1)
        if prob > threshold:
            points += np.random.normal(0, 0.02, size=points.shape)
        return points
    