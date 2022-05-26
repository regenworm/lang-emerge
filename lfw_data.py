import torch
import torchvision
import matplotlib.pyplot as plt


if __name__ == '__main__':
    lfw_data = torchvision.datasets.LFWPeople(
        'data', 'train', 'original', torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(lfw_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
    for img, target in data_loader:
        # img: (batch_size, x channels x width x height)
        # target: (batch_size) -> target identity as idx
        print(img.size(), target.size())
        plt.imshow(img[0, :, :, :].squeeze().permute([1, 2, 0]))
        plt.show()
        break
