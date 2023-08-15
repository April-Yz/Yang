## 你好

```python
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.kid import KernelInceptionDistance
import shutil

dataset_path = "F:\program1\dalle\FIDKID\DATA\Standard\Food\\"
#     #print(dataset_path)
#     for small in smalls:
#         if craft.split('_')[1] == small.split(' ')[0]:
fake_path =  "F:\program1\dalle\FIDKID\DATA\DALLE_DATA\Food\\"
image_paths = sorted([os.path.join(dataset_path, x) 
                                  for x in os.listdir(dataset_path)])
real_images = [np.array(Image.open(path).convert("RGB")) 
                           for path in image_paths]

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

real_images = torch.cat([preprocess_image(image) for image in real_images])
print(real_images.shape)
# torch.Size([10, 3, 256, 256])

image_paths = sorted([os.path.join(fake_path, x) for x in os.listdir(fake_path)] )
fake_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
fake_images = torch.cat([preprocess_image(image) for image in fake_images])
print(fake_images.shape)

"""k(x,y) = (\gamma * x^T y + coef)^{degree}
                1. normalize=True float``类型并且值在``[0,1]``范围内
                `False``图像的类型为`uint8`` [0,255]
                 所有图像将被调整为299 x 299，这是原始训练数据的大小。的boolian标志``real``确定图像是否应该更新真实分布的统计信息或假的分布。
                 ``kid_mean`` (:class:`~torch.Tensor`):包含子集上均值的浮点标量张量
                - ``kid_std`` (:class:`~torch.Tensor`):包含子集上均值的浮点标量张量
                subset 用于计算平均值和标准偏差得分的子集的数量
                subset_size:每个子集中随机抽取的样本数量
                degree:多项式核函数的次数
                gamma:多项式核函数的尺度长度。如果设置为``None``，将自动设置为特征大小
                coef:多项式核函数中的偏置项。
                reset_real_features:是否重置真实特征。
                """
kid = KernelInceptionDistance(normalize=True,subsets=1,subset_size=50)
# kid = KernelInceptionDistance(subset_size=60)
kid.update(real_images, real=True)
kid.update(fake_images, real=False)
kid_mean, kid_std = kid.compute()
print(f"KID:|||: (mean:{kid_mean},std:{kid_std})")
#F:\program1\dalle\FIDKID\DATA\Standard\Food
```

