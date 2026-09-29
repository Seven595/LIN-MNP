import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import numpy as np
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import random

class SKIN(torch.utils.data.Dataset):
    def __init__(self, train=True, val=False, gene_list=None, ds=None, sr=False, fold=0):
        super(SKIN, self).__init__()

        self.dir = './data/cscc/GSE144240_RAW/'
        self.precomputed_features_dir = './data/cscc/preprocessed_data/precomputed_features'
        self.cell_type_dir = './data/cscc/preprocessed_data/spots_type'
        self.r = 224 // 2

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)

        gene_list = list(np.load('./data/cscc/skin_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        self.train = train
        self.sr = sr

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}
        
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in self.meta_dict.items()}
        
        self.cell_type_dict = {}
        self.cell_type_mapping = {
            'nolabe': 0,
            'necros': 1,
            'neopla': 2,
            'inflam': 3,
            'connec': 4,
            'no-neo': 5
        }
        for name in names:
            cell_type_path = os.path.join(self.cell_type_dir, name, 'summary_spot_types.tsv')
            if os.path.exists(cell_type_path):
                df = pd.read_csv(cell_type_path, sep='\t')
                self.cell_type_dict[name] = df
            else:
                print(f"Warning: Cell type file not found for {name}")
                self.cell_type_dict[name] = None

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        
        self.id_dict = {i: m.index.values for i, m in self.meta_dict.items()}
        
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]
        spot_id = self.id_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        slice_name = self.id2name[i]
        feature_path = os.path.join(
            self.precomputed_features_dir, 
            slice_name, 
            f"{spot_id}.npy"
        )
        
        if not os.path.exists(feature_path):
            x, y = center
            alt_feature_path = os.path.join(
                self.precomputed_features_dir, 
                slice_name, 
                f"{int(x)}x{int(y)}.npy"
            )
            if os.path.exists(alt_feature_path):
                feature_path = alt_feature_path
            else:
                print(f"Warning: Feature file not found for {feature_path} and {alt_feature_path}")
                image_features = np.zeros(1024)
                np.save(feature_path, image_features)
        
        image_features = np.load(feature_path)

        item["image_features"] = torch.tensor(image_features, dtype=torch.float32)
        item["position"] = loc
        item["expression"] = exp
        
        slice_name = self.id2name[i]
        spot_id_str = spot_id
        cell_type = 0
        
        if self.cell_type_dict.get(slice_name) is not None:
            df = self.cell_type_dict[slice_name]
            if spot_id_str in df['spot'].values:
                cell_type_str = df[df['spot'] == spot_id_str]['type'].values[0]
                cell_type = self.cell_type_mapping.get(cell_type_str, 0)
        
        item["cell_type"] = torch.tensor(cell_type, dtype=torch.long)
        
        if not self.train:
            item["center"] = torch.Tensor(center)
            
        return item

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self, name):
        path = glob.glob(self.dir + '*' + name + '.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = glob.glob(self.dir + '*' + name + '_stdata.tsv')[0]
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = glob.glob(self.dir + '*spot*' + name + '.tsv')[0]
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class HERDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super().__init__()
        self.cnt_dir = './data/her2st/data/ST-cnts'
        self.precomputed_features_dir = './data/her2st/preprocessed_data/precomputed_features'
        self.pos_dir = './data/her2st/data/ST-spotfiles'
        self.lbl_dir = './data/her2st/data/ST-pat'
        self.cell_type_dir = './data/her2st/preprocessed_data/spots_type'
        self.r = 224 // 2
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]

        self.train = train

        samples = names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))
        if train:
            names = tr_names
        else:
            names = te_names
            self.meta_dict = {i: self.get_meta(i) for i in names}
            self.names = te_names
            self.label = {i: None for i in self.names}
            self.lbl2id = {
                'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
                'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
            }
            if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
                idx = self.meta_dict[self.names[0]].index
                lbl = self.lbl_dict[self.names[0]]
                lbl = lbl.loc[idx, :]['label'].values
                self.label[self.names[0]] = lbl

        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.center_dict = {
            i: m[['pixel_x', 'pixel_y']].values.astype(int)
            for i, m in self.meta_dict.items()
        }

        self.cell_type_dict = {}
        self.cell_type_mapping = {
            'nolabe': 0,
            'necros': 1,
            'neopla': 2,
            'inflam': 3,
            'connec': 4,
            'no-neo': 5
        }
        for name in names:
            cell_type_path = os.path.join(self.cell_type_dir, name, 'summary_spot_types.tsv')
            if os.path.exists(cell_type_path):
                df = pd.read_csv(cell_type_path, sep='\t')
                self.cell_type_dict[name] = df
            else:
                print(f"Warning: Cell type file not found for {name}")
                self.cell_type_dict[name] = None

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        
        self.id_dict = {i: m.index.values for i, m in self.meta_dict.items()}
        
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        exp = self.exp_dict[self.id2name[i]][idx]
        spot_id = self.id_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        slice_name = self.id2name[i]
        feature_path = os.path.join(
            self.precomputed_features_dir, 
            slice_name, 
            f"{spot_id}.npy"
        )
        
        if not os.path.exists(feature_path):
            x, y = loc
            alt_feature_path = os.path.join(
                self.precomputed_features_dir, 
                slice_name, 
                f"{int(x)}x{int(y)}.npy"
            )
            if os.path.exists(alt_feature_path):
                feature_path = alt_feature_path
            else:
                print(f"Warning: Feature file not found for {feature_path} and {alt_feature_path}")
                image_features = np.zeros(1024)
                np.save(feature_path, image_features)
        
        image_features = np.load(feature_path)

        feature_path_112 = os.path.join(
            self.precomputed_features_dir_112, 
            slice_name, 
            f"{spot_id}.npy"
        )
        
        if not os.path.exists(feature_path_112):
            x, y = loc
            alt_feature_path_112 = os.path.join(
                self.precomputed_features_dir_112, 
                slice_name, 
                f"{int(x)}x{int(y)}.npy"
            )
            if os.path.exists(alt_feature_path_112):
                feature_path_112 = alt_feature_path_112
            else:
                print(f"Warning: Feature file not found for {feature_path_112} and {alt_feature_path_112}")
                image_features_112 = np.zeros(1024)
                np.save(feature_path_112, image_features_112)
        
        image_features_112 = np.load(feature_path_112)
        
        item["image_features"] = torch.tensor(image_features, dtype=torch.float32)
        item["image_features_112"] = torch.tensor(image_features_112, dtype=torch.float32)
        item["position"] = loc
        item["expression"] = exp
        
        slice_name = self.id2name[i]
        spot_id_str = spot_id
        cell_type = 0
        
        if self.cell_type_dict.get(slice_name) is not None:
            df = self.cell_type_dict[slice_name]
            if spot_id_str in df['spot'].values:
                cell_type_str = df[df['spot'] == spot_id_str]['type'].values[0]
                cell_type = self.cell_type_mapping.get(cell_type_str, 0)
        
        item["cell_type"] = torch.tensor(cell_type, dtype=torch.long)
        
        if not self.train:
            center = self.center_dict[self.id2name[i]][idx]
            item["center"] = torch.Tensor(center)
            
        return item

    def __len__(self):
        return self.cumlen[-1]

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + 'lbl' + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class SliceBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.slice_indices = {}
        
        start = 0
        for i, length in enumerate(dataset.lengths):
            slice_name = dataset.id2name[i]
            end = start + length
            self.slice_indices[slice_name] = list(range(start, end))
            start = end
    
    def __iter__(self):
        slice_names = list(self.slice_indices.keys())
        random.shuffle(slice_names)
        
        for slice_name in slice_names:
            indices = self.slice_indices[slice_name]
            random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
    
    def __len__(self):
        total = 0
        for indices in self.slice_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


class TenxDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path,
                 reduced_mtx_path, precomputed_features_dir,
                 cell_type_path=None, train=True):

        self.train = train
        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T
        self.precomputed_features_dir = precomputed_features_dir

        self.cell_type_mapping = {
            'nolabe': 0, 'necros': 1, 'neopla': 2,
            'inflam': 3, 'connec': 4, 'no-neo': 5
        }
        self.cell_type_dict = None
        if cell_type_path is not None and os.path.exists(cell_type_path):
            df = pd.read_csv(cell_type_path, sep="\t")
            self.cell_type_dict = dict(zip(df['spot'], df['type']))
        else:
            print("Warning: cell_type_path not found, default cell_type=0")

    def __len__(self):
        return len(self.barcode_tsv)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0]

        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]
        position = torch.tensor([v1, v2], dtype=torch.float32)

        feature_path = os.path.join(self.precomputed_features_dir, f"{barcode}.npy")
        if not os.path.exists(feature_path):
            image_features = np.zeros(1024)
        else:
            image_features = np.load(feature_path)

        expression = self.reduced_matrix[idx, :]

        cell_type = 0
        if self.cell_type_dict is not None and barcode in self.cell_type_dict:
            cell_type_str = self.cell_type_dict[barcode]
            cell_type = self.cell_type_mapping.get(cell_type_str, 0)

        item["image_features"] = torch.tensor(image_features, dtype=torch.float32)
        item["expression"] = torch.tensor(expression, dtype=torch.float32)
        item["position"] = position
        item["cell_type"] = torch.tensor(cell_type, dtype=torch.long)
        return item

class ViT_HBCIST(torch.utils.data.Dataset):
    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super().__init__()
        self.cnt_dir = r'./data/Hbcist_THItoGene/ST-cnts'
        self.img_dir = r'./data/Hbcist_THItoGene/ST-imgs'
        self.pos_dir = r'./data/Hbcist_THItoGene/ST-spotfiles'
        self.lbl_dir = r'./data/Hbcist_THItoGene/ST-pat/lbl'
        self.r = 224 // 2
        gene_list = list(np.load(r'./data/Hbcist_THItoGene/target_list.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:10] for i in names]

        self.train = train

        samples = names
        te_names = []
        for i in fold:
            te_names.append(samples[i])
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
            names = tr_names
        else:
            self.names = te_names
            names = te_names

            self.meta_dict = {i: self.get_meta(i) for i in names}
            self.names = te_names
            self.label = {i: None for i in self.names}
            self.lbl2id = {
                'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
                'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
            }

        print(names)
        print("Loading imgs ...")
        self.img_dict = {i: self.get_img(i) for i in names}
        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}

        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            return item

        else:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return item

    def __len__(self):
        return self.cumlen[-1]

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_coords.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '_exp.csv'
        df = pd.read_csv(path, index_col=0)
        return df

    def get_img(self, name):
        path = self.img_dir + '/' + name + '.jpg'
        print(path)
        im = Image.open(path)
        return im

    def get_lbl(self, name):
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)