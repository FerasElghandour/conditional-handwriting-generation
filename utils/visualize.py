"""Labeled grids with the same fixed intensity scale for every model."""
import math
import matplotlib.pyplot as plt

def show_batch(images, labels, n=16):
    n=min(n,len(images)); columns=min(10,math.ceil(math.sqrt(n))); rows=math.ceil(n/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(columns,rows),squeeze=False)
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if i<n:
            ax.imshow(images[i].detach().cpu().squeeze(),cmap='gray',vmin=-1,vmax=1)
            ax.set_title(f"Label {int(labels[i])}",fontsize=8)
    plt.tight_layout(); plt.show()

def class_grid(images, labels, per_class=5, title='Conditioned samples'):
    fig,axes=plt.subplots(per_class,10,figsize=(12,1.25*per_class),squeeze=False)
    for digit in range(10):
        selected=images[labels==digit][:per_class]
        if len(selected)<per_class: raise ValueError('Insufficient images for class')
        for row,image in enumerate(selected):
            axes[row,digit].imshow(image.detach().cpu().squeeze(),cmap='gray',vmin=-1,vmax=1)
            axes[row,digit].axis('off')
            if row==0: axes[row,digit].set_title(f'Class {digit}')
    fig.suptitle(title); plt.tight_layout(); plt.show()

def plot_real_cgan_diffusion(real_images,real_labels,cgan_images,cgan_labels,diff_images,diff_labels):
    fig,axes=plt.subplots(3,10,figsize=(13,4))
    for row,(name,images,labels) in enumerate([('Real',real_images,real_labels),('cGAN',cgan_images,cgan_labels),('Diffusion',diff_images,diff_labels)]):
        for digit in range(10):
            image=images[labels==digit][0]
            axes[row,digit].imshow(image.detach().cpu().squeeze(),cmap='gray',vmin=-1,vmax=1)
            axes[row,digit].set_xticks([]); axes[row,digit].set_yticks([])
            if row==0: axes[row,digit].set_title(f'Class {digit}')
            if digit==0: axes[row,digit].set_ylabel(name)
    fig.suptitle('First sample per requested class; no selection for appearance')
    plt.tight_layout(); plt.show()
