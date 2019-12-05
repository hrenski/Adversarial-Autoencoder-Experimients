#! /usr/bin/env python

import numpy as np

import torch
import torchvision

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imageio
from pygifsicle import optimize

font = {'weight': 'normal', 'size': 16}

def plot_tensor(img1, img2 = None, cmap = "Greys", nrow = 8, figsize=(20, 10), titles = None):
    if img2 is not None:

        fig, ax  = plt.subplots(figsize=figsize, nrows = 1, ncols = 2)
        tmp1 = ax[0].imshow(torchvision.utils.make_grid(img1.detach().cpu().reshape(-1, img1.size(1), img1.size(2), img1.size(3)), padding = 8, nrow = nrow).permute(1, 2, 0), cmap=cmap)
        tmp2 = ax[1].imshow(torchvision.utils.make_grid(img2.detach().cpu().reshape(-1, img2.size(1), img2.size(2), img2.size(3)), padding = 8, nrow = nrow).permute(1, 2, 0), cmap=cmap)

        for a in ax:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(tmp2, cax=cax)

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(tmp1, cax=cax)
        
        if titles is not None:
            ax[0].set_title(titles[0], fontsize = font['size'])
            
            if len(titles) > 1:
                ax[1].set_title(titles[1], fontsize = font['size'])

    else:
        fig, ax  = plt.subplots(figsize=figsize, nrows = 1, ncols = 1)

        tmp = ax.imshow(torchvision.utils.make_grid(img1.detach().cpu().reshape(-1, img1.size(1), img1.size(2), img1.size(3)), padding = 8, nrow = nrow, scale_each = True, normalize=True).permute(1, 2, 0), cmap=cmap, vmin = 0., vmax = 1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(tmp, cax=cax)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if titles is not None:
            ax.set_title(titles[0], fontsize = font['size'])
       

    plt.tight_layout()
    plt.show()
        
def plot_labels(data, lbl, pred = None, lbl_dict = None, fig_shape = (8,8), figsize = (12,10), up_fctr = 3):
    fig, ax = plt.subplots(figsize = figsize, nrows = fig_shape[0], ncols = fig_shape[1])

    transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                    torchvision.transforms.Resize(data.size(-1) * up_fctr, interpolation = 3),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Lambda(lambda x : x.permute(1,2,0))])

    for i in range(ax.size):
        ax[np.unravel_index(i, fig_shape)].imshow(transform(data[i]).detach().cpu().numpy())
        ax[np.unravel_index(i, fig_shape)].get_xaxis().set_visible(False)
        ax[np.unravel_index(i, fig_shape)].get_yaxis().set_visible(False)
        if pred is not None:
            ax[np.unravel_index(i, fig_shape)].set_title("T: {:d}, P: {:d}".format(lbl[i], pred[i]), fontsize = font['size'])
        else:
            ax[np.unravel_index(i, fig_shape)].set_title("T: {:d}".format(lbl[i]), fontsize = font['size'])


    if pred is not None:
        fig.text(1.1, 0.90, "T: Target\nP: Predicted", backgroundcolor = "slategray", fontsize = font['size'])
    else:
        fig.text(1.1, 0.90, "T: Target", backgroundcolor = "slategray", fontsize = font['size'])
        
    if lbl_dict is not None:
        txt = '\n'.join(["{:d}: {:s}".format(val, key) for key,val in lbl_dict.items()])
        fig.text(1.1, 0.4, txt, backgroundcolor = "slategray", fontsize = font['size'])
    plt.tight_layout(pad=0, h_pad=0.5, w_pad=0.5, rect=None)
    plt.show()
        
def view_2dDistro(info1, info2 = None, figsize = (10,9)):
    colors = [ 'red', 'green', 'orange', 'limegreen', 'blue', 'brown', 'salmon', 'cyan',
              'yellow', 'magenta', 'slategray']
    n_bin = len(colors)

    cmap_name = 'my_list'
    tst_cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)

    (coord1, lbl1) = info1
    
    if info2 is not None:
        coord2,lbl2 = info2
        
        fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = figsize, sharey = True)
        sc1= ax[0].scatter(coord1[:,0], coord1[:,1], s = 1, c = lbl1, cmap = tst_cmap, norm = mpl.colors.BoundaryNorm(np.arange(n_bin + 1), n_bin))
        sc2 = ax[1].scatter(coord2[:,0], coord2[:,1], s = 1, c = lbl2, cmap = tst_cmap, norm = mpl.colors.BoundaryNorm(np.arange(n_bin + 1), n_bin))

        ax[0].set_title("Posterior Distribution", fontsize = font['size'])
        ax[1].set_title("Prior Distribution", fontsize = font['size'])

        ax[0].grid()
        ax[1].grid()

        ax[0].set_xlim([-5,5])
        ax[1].set_xlim([-5,5])
        ax[0].set_ylim([-5,5])

        cbar = plt.colorbar(sc2, ticks = np.arange(n_bin + 1))
    else:
        fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, sharey = True)
        sc = ax.scatter(coord1[:,0], coord1[:,1], s = 1, c = lbl1, cmap = tst_cmap, norm = mpl.colors.BoundaryNorm(np.arange(n_bin + 1), n_bin))

        ax.set_title("Posterior Distribution", fontsize = font['size'])
        ax.grid()

        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])

        cbar = plt.colorbar(sc, ticks = np.arange(n_bin + 1))
        
    plt.tight_layout()
    plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = ax.figure.colorbar(im, ax=ax, cax = cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #cbar = plt.colorbar(im, cax=cax)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontdict = font)
    ax.set_yticklabels(row_labels, fontdict = font)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return text

class gifMaker(object):
    def __init__(self, encoder, decoder, frames = 32, slowness = None):
    
        self.__encoder = encoder
        self.__decoder = decoder
        
        self.device = torch.device("cuda:0" if next(self.__encoder.parameters()).is_cuda else "cpu")

        self.__encoder = self.__encoder.to(self.device)
        self.__decoder = self.__decoder.to(self.device)
        
        self.frames = frames
        self.slowness = slowness
        
    @property
    def frames(self):
        return self.__frames

    @frames.setter
    def frames(self, value):
        self.__frames = value
        
        if hasattr(self, "slowness"):
            self.slowness = min(self.slowness, self.frames)
        else:
            self.slowness = self.frames

    @property
    def slowness(self):
        return self.__slowness

    @slowness.setter
    def slowness(self, value):
        if value is None:
            value = self.frames

        value = int((value - 1) / 2.) * 2 + 1
        self.__slowness = min(value, self.frames - 2 if (self.frames % 2) else self.frames - 1)
        
        self.__update_transition()
        
    @property
    def transition(self):
        return self.__transition

    def __update_transition(self):
        tmp = np.zeros(self.frames, dtype = 'f4')
        tmp[tmp.size // 2:] = 1.

        self.__transition = np.square(np.cos(nd.uniform_filter1d(tmp, self.__slowness, mode='nearest') * np.pi/2))
    
    def gif(self, gif_fn, img_tnsr, reverse = True, up_fctr = 1, duration = 0.1):

        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None), torchvision.transforms.Resize(img_tnsr[0].size(-1) * up_fctr, 3)])

        img_tnsr = img_tnsr.to(self.device)
        
        if reverse:
            path_parm = np.r_[self.__transition, self.__transition[::-1]]
        else:
            path_parm = self.__transition
        
        with imageio.get_writer(gif_fn, mode='I', duration = duration) as writer:
            for i in range(img_tnsr.size(0) - 1):
                vtx = img_tnsr[[i,  i+1]]
                vtx_enc = encoder(vtx)
                for j in trange(path_parm.size, miniters = 1, 
                                leave = False, ncols = 100, 
                                desc = "{:3d}/{:3d}".format(i+1,img_tnsr.size(0)),
                                position=0):
                                            
                    tmp = vtx_enc[0] * path_parm[j] + (1 - path_parm[j]) * vtx_enc[1]
                    writer.append_data(np.array(transform(self.__decoder(tmp).view(3,64,64))))

        optimize(gif_fn)
        
    def grid_gif(self, gif_fn, img_tnsr, idx, reverse = True, up_fctr = 3, duration = 0.1, nrow = 4):

        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None), 
                                                    torchvision.transforms.Resize(img_tnsr[0].size(-1) * up_fctr, 3),
                                                    torchvision.transforms.ToTensor()])
        img_tnsr = img_tnsr.to(self.device)
        
        if reverse:
            path_parm = np.r_[self.__transition, self.__transition[::-1]]
        else:
            path_parm = self.__transition

        with imageio.get_writer(gif_fn, mode='I', duration=duration) as writer:
            for i in range(path_parm.size):
                path_enc = torch.empty(len(idx), self.__encoder.num_dim, dtype = torch.float32, device = self.device)
                
                for j in trange(len(idx), miniters = 1, leave = False, ncols = 100, 
                                desc = "{:3d}/{:3d}".format(i + 1, path_parm.size), position=0):
                    vtx = img_tnsr[idx[j]]
                    vtx_enc = self.__encoder(vtx)

                    path_enc[j] = vtx_enc[0] * path_parm[i] + (1 - path_parm[i]) * vtx_enc[1]

                path_dc = self.__decoder(path_enc)

                dc_up = torch.empty(path_dc.size(0), path_dc.size(1), path_dc.size(2) * up_fctr, path_dc.size(3) * up_fctr, dtype = torch.float, device = self.device)

                for k in range(path_dc.size(0)):
                    dc_up[k] = transform(path_dc[k])

                img_grid = torchvision.utils.make_grid(dc_up.detach(), padding = 8, nrow = 4)#.permute(1, 2, 0)
                writer.append_data(np.array(torchvision.transforms.ToPILImage(mode=None)(img_grid)))

        optimize(gif_fn)
        
