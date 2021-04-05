import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import imageio

from IQA_pytorch import SSIM, MS_SSIM, CW_SSIM, GMSD, LPIPSvgg, DISTS, NLPD, FSIM, VSI, VIFs, VIF, MAD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ref_path  = 'images/62.jpg'
pred_path = 'images/sn_62.jpg' 

model = SSIM(channels=3).to(device)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
ref_img = Image.open(ref_path).convert("RGB")
ref = transform(ref_img).unsqueeze(0)
ref = Variable(ref.float().to(device), requires_grad=False)

pred_img = Image.open(pred_path).convert("RGB")
pred = transform(pred_img).unsqueeze(0)
pred = Variable(pred.float().to(device), requires_grad=True)

# pred = torch.rand_like(pred)
# pred.requires_grad_(True)
# pred_img = pred.squeeze().data.cpu().numpy().transpose(1, 2, 0)
# pred_img = pred.squeeze().data.cpu().numpy().transpose(1, 2, 0)

model.eval()
fig = plt.figure(figsize=(4,1.5),dpi=300)
plt.subplot(131)
plt.imshow(pred_img)
plt.title('initial',fontsize=6)
plt.axis('off')
plt.subplot(133)
plt.imshow(ref_img)
plt.title('reference',fontsize=6)
plt.axis('off')

lr = 0.005
optimizer = torch.optim.Adam([pred], lr=lr)
iteration = []
dist_list = []

for i in range(1, 20001):
    iteration.append(i) 
    dist = model(pred, ref)
    dist_list.append(dist)
    optimizer.zero_grad()
    dist.backward()
    # torch.nn.utils.clip_grad_norm_([pred], 1)
    optimizer.step()
    pred.data.clamp_(min=0,max=1)

    # print(dist.item())
    # break
    
    if i % 50 == 0:
        pred_img = pred.squeeze().data.cpu().numpy().transpose(1, 2, 0)
        plt.subplot(132)       
        plt.imshow(np.clip(pred_img, 0, 1))        
        plt.title('iter: %d, dists: %.3g' % (i, dist.item()),fontsize=6)
        plt.axis('off')
        plt.pause(1)
        plt.cla()

        if i == 500 or i == 2000 or i == 10000 or i == 20000: 
            imageio.imwrite('results/ssim/' + str(i) + '.jpg',pred_img)

    if (i+1) % 2000 == 0:
        lr = max(1e-4, lr*0.5)
        optimizer = torch.optim.Adam([pred], lr=lr)

## two places to be changed
# plt.close()
# plt.plot(iteration, dist_list, label = 'SSIM')
# plt.xlabel('iteration')
# plt.ylabel('dist')
# plt.legend()
# plt.savefig('results/ssim/dist.jpg')
# plt.show()
