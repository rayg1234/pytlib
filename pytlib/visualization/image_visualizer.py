import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# Singleton class for visualization images during training, similiar to the logger
class ImageVisualizer:
    instance = None

    class __ImageVisualizer:
        def __init__(self):
            # raise exception here
            self.cur_images = dict()
            self.max_cols = 8

        def dump_image(self,output_file,display=False,block=True,save=True):
            rows = int(math.ceil(float(len(self.cur_images)) / self.max_cols))
            cols = self.max_cols if len(self.cur_images) > self.max_cols else len(self.cur_images)
            fig,axes = plt.subplots(rows,cols)
            # fig.subplots_adjust(hspace=0.5, wspace=1.0)
            fig.canvas.set_window_title('Visualizations')
            # print rows, cols
            if rows==1 and cols==1:
                axes = [[axes]]
            elif rows==1 and cols>1:
                axes = [axes]

            for i in range(0,rows):
                for j in range(0,cols):
                    axes[i][j].axis('off')

            for i,(key,image) in enumerate(sorted(self.cur_images.iteritems())):
                (r,c) = np.unravel_index(i, (rows,cols))
                ax = axes[r][c]
                ax.set_title(key)
                ax.set_xticklabels([])
                ax.set_yticklabels([])                
                image.visualize(axes=ax,display=False)

            if save:
                fig.savefig(output_file)

            if display:
                plt.show(block=block)

        def set_image(self,pt_image,key):
            self.cur_images[key]=pt_image

    def __init__(self):
        if not ImageVisualizer.instance:
            ImageVisualizer.instance = ImageVisualizer.__ImageVisualizer()

    def __getattr__(self,name):
        return getattr(self.instance,name)
