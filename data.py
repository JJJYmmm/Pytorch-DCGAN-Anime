from torch.utils.data import DataLoader
from torchvision import utils,datasets,transforms

class ReadData():
    def __init__(self,data_path,image_size=64):
        self.root = data_path
        self.image_size = image_size
        self.dataset = self.getdataset()
    
    def getdataset(self):
        dataset = datasets.ImageFolder(
                root=self.root,
                transform=transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            )
        print(f'Total size of dataset:{len(dataset)}')
        return dataset
    
    def getdataloader(self,batch_size=128):
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        return dataloader
    
if __name__ == '__main__':
    dset = ReadData('./imgs')
    print("ok")
    dloader = dset.getdataloader()