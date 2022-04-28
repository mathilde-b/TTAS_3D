#!/usr/bin/env python3.8
import os
import re
import argparse
from pathlib import Path
from pprint import pprint
from functools import partial
from collections import namedtuple
from typing import Callable, Dict, List, Tuple
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from skimage.io import imread
from skimage.transform import resize
from matplotlib.colors import ListedColormap
from viewerr import get_image_lists
from utils import read_anyformat_image
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import ndimage
import matplotlib.patches as mpatches

# Based on torchvision, itself based on
# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                             'has_instances', 'ignore_in_eval', 'color'])

city_classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),

    CityscapesClass('gta5thing', 34, 34, 'void', 0, False, True, (255, 255, 255))
    # CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

mycmap = ListedColormap(["black","peru", "peru", "mediumpurple", "mediumpurple","darkorange","moccasin", "moccasin", "mediumpurple", "lightblue","black","peru",])

#mycmap = ListedColormap(["black","cornflowerblue", "cornflowerblue", "firebrick", "firebrick","darkorange","mediumaquamarine", "mediumaquamarine", "firebrick", "lemonchiffon","black","cornflowerblue",])

def intersection(lsts):
    for (i,lst) in enumerate(lsts):
        if i==0:
            lst_int = lst
        lst_int = list(set(lst_int) & set(lst))
        #print(lst_int)
    return lst_int


def extract(pattern: str, string: str) -> str:
    try:
        return re.match(pattern, string).group(1)
    except AttributeError:  # id not found
        return None


def display_item(axe, img: np.ndarray, mask: np.ndarray, contour: bool, cmap,
                 args):
    m = resize(mask, img.shape[:2], mode='constant', preserve_range=True)
    #print(np.unique(mask))
    # try:
    #     assert len(img.shape) == len(m.shape)
    # except AssertionError:
    #     # print(title)
    #     print(img.shape, m.shape)
    #     # raise

    #     # Some grayscale mask are sometimes loaded with 3 channel
    #     # m = m[:, :, 0]
    #     m = m[..., None]
    #     # img = np.moveaxis(img, -1, 0)

    #axe.imshow(np.flipud(ndimage.rotate(img, 270)), cmap="gray")
    axe.imshow(ndimage.rotate(img, 90), cmap="gray")
    #axe.imshow(img, cmap="gray")

    if contour:
        axe.contour(m, cmap=cmap)
    elif not args.ent_map:
        #axe.imshow(np.flipud(ndimage.rotate(m, 270)), cmap=cmap, alpha=args.alpha, vmin=0, vmax=args.C)
        axe.imshow((ndimage.rotate(m, 90)), cmap=cmap, alpha=args.alpha, vmin=0, vmax=args.C)
        #axe.imshow(m, cmap=cmap, alpha=args.alpha, vmin=0, vmax=args.C)
    else:
        #axe.imshow(m, alpha=1,cmap="jet",norm = mpl.colors.Normalize(vmin=0,vmax=176))
        mask=axe.imshow(ndimage.rotate(m, 90), alpha=0.1,cmap="rainbow",norm = mpl.colors.PowerNorm(gamma=0.3,vmin=0,vmax=176))
        forleg = axe.imshow(ndimage.rotate(m, 90), alpha=1)
        values = np.unique(m.ravel())
        #print(values)
        #print(min(values),'min values')
        #print(max(values),'max values')
        val_bins = np.linspace(min(values),max(values),30)
        bins = np.digitize(values,np.linspace(min(values),max(values),30))
        #print(np.unique(bins),'bins')
        # get the colors of the values, according to the
        # colormap used by imshow
        #colors = [forleg.cmap(forleg.norm(value)) for value in values]
        colors = [forleg.cmap(forleg.norm(val_bin)) for val_bin in val_bins]
        # create a patch (proxy artist) for every color
        #patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in range(len(values))]
        patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=round(val_bins[i],1))) for i in range(len(np.unique(val_bins)))]
        #print(len(patches),'len patches')
        # put those patched as legend-handles into the legend
        #plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
        #axe.legend(mask)
    axe.axis('off')


def display(background_names: List[str], segmentation_names: List[List[str]],
            indexes: List[int], column_title: List[str], row_title: List[str],
            crop: int, contour: bool, remap: Dict, fig=None, args=None) -> None:
    if not fig:
        fig = plt.figure()
    grid = gridspec.GridSpec(len(indexes) + args.legend, len(segmentation_names),
                             height_ratios=[((0.9 + 0.1 * ~args.legend) / len(indexes))
                                            for _ in range(len(indexes))] + ([0.1] if args.legend else []))
    grid.update(wspace=0.025, hspace=0.05)
    names: List[str]
    if args.cmap == 'cityscape':
        colors = [tuple(c / 255 for c in e.color) for e in city_classes]
        names = [e.name for e in city_classes]

        cmap = ListedColormap(colors, 'cityscape')
    else:
        # cmap = args.cmap
        cmap2 = matplotlib.cm.get_cmap(args.cmap)
        cmap = matplotlib.cm.get_cmap(mycmap)
        #print(len(cmap.colors),len(cmap2.colors))
        names = list(map(str, range(args.C))) if not args.class_names else args.class_names
        assert len(names) == args.C
    #print([cmap(v / args.C) for v in range(args.C)])
    if args.legend:
        ax = plt.subplot(grid[-1, :])

        ax.bar(list(range(args.C)), [1] * args.C,
               tick_label=names,
               color=[cmap(v / args.C) for v in range(args.C)])

        ax.set_xticklabels(names, rotation=60)
        ax.set_xlim([-0.5, args.C - 0.5])
        ax.get_yaxis().set_visible(False)
        ax.set_title("Legend")
    
    for i, idx in enumerate(indexes):
        #print(i)
        img: np.ndarray = read_anyformat_image(background_names[idx])

        if crop > 0:
            img = img[crop:-crop, crop:-crop]

        for j, names in enumerate(segmentation_names):
            ax_id = len(segmentation_names) * i + j
            # axe = grid[ax_id]
            axe = plt.subplot(grid[ax_id])

            seg: np.ndarray = read_anyformat_image(names[idx])

            #print(np.unique(seg))
            if crop > 0:
                seg = seg[crop:-crop, crop:-crop]
            if remap:
                for k, v in remap.items():
                    seg[seg == k] = v

            display_item(axe, img, seg, contour, cmap, args)

            #if j == 0:
                #print(row_title[idx])
                #axe.text(-30, seg.shape[1] // 2, row_title[idx], rotation=90,
                #         verticalalignment='center', fontsize=14)
            if i == 0:
                axe.set_title(column_title[j])

        fig.show()
        #print("scp -r AP50860@koios.logti.etsmtl.ca:../../../data/users/mathilde/ccnn/CDA/"+args.result_fold+'/'+'__'.join([row_title[idx] for idx in indexes])+'.png ./seg/whs/')
        #print("scp -r AP50860@koios.logti.etsmtl.ca:../../../data/users/mathilde/ccnn/CDA/"+args.result_fold+' ./seg/whs/')
        plt.show()
        plt.savefig(args.result_fold+'/'+'__'.join([row_title[idx] for idx in indexes])+'.png',dpi=200)
        plt.close()


def get_image_lists(img_source: str, folders: List[str], id_regex: str) -> Tuple[List[str], List[List[str]], List[str]]:
    path_source: Path = Path(img_source)
    background_names: List[str] = sorted(map(str, path_source.glob("*")))
    print("background_names[1:10]",background_names[1:10])

    segmentation_names: List[List[str]] = [sorted(map(str, Path(folder).glob("*"))) for folder in folders]

    # do intersecction of lists
    back_ex = [extract(id_regex,bg) for bg in background_names if extract(id_regex,bg) is not None]
    seg_ex = [[extract(id_regex, sn) for sn in sl if extract(id_regex, sn) is not None] for sl in segmentation_names]
    print(id_regex,"id_regex",back_ex[1:10],'backex')
    print(seg_ex[0][1:10],'segex')
    print("len seg_ex[0]", len(seg_ex[0]))
    print("len seg_ex + back_ex", len([back_ex]+[s for s in seg_ex]))
    print(back_ex[1:10],seg_ex[0][1:10])
    intersec= intersection([back_ex]+[s for s in seg_ex])
    print(len(intersec))

    background_names = [bg for bg in background_names if extract(id_regex,bg) is not None and extract(id_regex,bg) in intersec]
    segmentation_names = [[sn for sn in sl if extract(id_regex, sn) is not None and extract(id_regex,sn) in intersec] for sl in segmentation_names]

    print("background_names[:10]",background_names[:10])
    for s in segmentation_names:
        print("segmentation_names[:10]",s[:10])

    ids: List[str] = [extract(id_regex,bg) for bg in background_names]

    for names, folder in zip(segmentation_names, folders):
         try:
             assert(len(background_names) == len(names))
    #         assert(ids == list(map(extracter, names)))
         except AssertionError:
             print(f"Error verifying content for folder {folder}")
             print(f"Background folder '{img_source}': {len(background_names)} imgs")
             pprint(background_names[:10])
             print(f"Folder '{folder}': {len(names)} imgs")
             pprint(names[:10])

    return background_names, segmentation_names,ids


class EventHandler(object):
    def __init__(self, order: List[int], n: int, draw_function: Callable, fig):
        self.order: List[int] = order
        self.draw_function: Callable = draw_function
        self.n = n
        self.i = 0
        self.fig = fig

    def __call__(self, event):
        if event.button == 1:  # next
            self.i += 1
        elif event.button == 3:  # prev
            self.i -= 1

        a = self.i * self.n

        self.redraw(a)

    def redraw(self, a):
        self.fig.clear()

        idx: List[int] = self.order[a:a + self.n]

        self.draw_function(idx, fig=self.fig)

        self.fig.canvas.draw()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display the requested data.")
    parser.add_argument("--img_source", type=str, required=True,
                        help="The folder containing the images (background).")
    parser.add_argument("--result_fold", type=str,required=True)
    parser.add_argument("-n", type=int, default=3,
                        help="The number of images to sample per window.")
    parser.add_argument("--seed", type=int, default=0,
                        help="The seed for the number generator. Used to sample the images. \
                             Useful to reproduce the same outputs between runs.")
    parser.add_argument("--crop", type=int, default=0,
                        help="The number of pixels to remove from each border")
    parser.add_argument("-C", type=int, default=2,
                        help="Number of city_classes. Useful when not all of them appear on each images.")

    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--id_regex", type=str, default=".*/(.*).png",
                        help="The regex to extract the image id from the images names \
                             Required to match the images between them.")
    parser.add_argument("folders", type=str, nargs='*',
                        help="The folder containing the source segmentations.")
    parser.add_argument("--display_names", type=str, nargs='*',
                        help="The display name for the folders in the viewer")
    parser.add_argument("--class_names", type=str, nargs='*')
    parser.add_argument("--remap", type=str, default="{}",
                        help="Remap some mask values if needed. Useful to suppress some city_classes.")

    parser.add_argument("--no_contour", action="store_true",
                        help="Do not draw a contour but a transparent overlap instead.")
    parser.add_argument("--legend", action="store_true",
                        help="When set, display the legend of the colors at the bottom")

    parser.add_argument("--cmap", default='rainbow', choices=list(cm.datad.keys()) + ['cityscape'])
    parser.add_argument("--ent_map", type=bool,default=False)
    args = parser.parse_args()

    return args


def main() -> None:
    args: argparse.Namespace = get_args()
    np.random.seed(args.seed)
    if not os.path.isdir(args.result_fold):
        os.mkdir(args.result_fold)
    background_names: List[str]
    segmentation_names: List[List[str]]
    ids: List[str]
    background_names, segmentation_names, ids = get_image_lists(args.img_source, args.folders, args.id_regex)

    if args.display_names is None:
        display_names = [f for f in args.folders]
    else:
        assert len(args.display_names) == len(args.folders), (args.display_names, args.folders)
        display_names = args.display_names

    order: List[int] = list(range(len(background_names)))
    #order = np.random.permutation(order)
    order = np.array(order)
    draw_function = partial(display, background_names, segmentation_names,
                            column_title=display_names,
                            row_title=ids,
                            crop=args.crop,
                            contour=not args.no_contour,
                            remap=eval(args.remap),
                            args=args)

    fig = plt.figure()
    
    print("scp -r AP50860@phoebe.logti.etsmtl.ca:../../../data/users/mathilde/ccnn/CDA/TT/"+args.result_fold+' ./seg/')
    #event_handler = EventHandler(order, args.n, draw_function, fig)
    #fig.canvas.mpl_connect('button_press_event', event_handler)
    for a in range(0, len(background_names), args.n):
        idx: List[int] = order[a:a + args.n]
        draw_function(order[idx], fig=fig)
    #plt.show()


if __name__ == "__main__":
    main()
