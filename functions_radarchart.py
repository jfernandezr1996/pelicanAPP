import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class ComplexRadar():
    
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        mpl.style.use('seaborn')
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        lines, labels = axes[0].set_thetagrids(angles, labels=variables)

        new_l = []
        a_l = []
        for label, angle in zip(labels, angles):
            x,y = label.get_position()
            lab = axes[0].text(x,y, label.get_text(), 
                               transform=label.get_transform(),
                               ha=label.get_ha(), va=label.get_va())
            if (angle <= 180):
                angle_rotate = angle - 29*np.pi
            else:
                angle_rotate = angle + 29*np.pi
            a_l.append(angle_rotate)
            lab.set_rotation(angle_rotate)
            new_l.append(lab)
        axes[0].set_xticklabels([])
    
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) for x in grid]
            gridlabel[0] = ""
            gridlabel[-1] = ""
            ax.set_rgrids(grid, labels=gridlabel, 
                          angle=angles[i], fontsize=7)
            ax.set_ylim(*ranges[i])
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        
    def preprocess_data(self, data):
        for d, (y1, y2) in zip(data[1:], self.ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
            x1, x2 = self.ranges[0]
            d = data[0]
            sdata = [d]
            for d, (y1, y2) in zip(data[1:], self.ranges[1:]):
                sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
        return sdata
        
    def plot(self, data, *args, **kw):
        sdata = self.preprocess_data(data)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=5)
        
    def fill(self, data, *args, **kw):
        sdata = self.preprocess_data(data)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    