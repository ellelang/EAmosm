import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.table import Table
#the_table.auto_set_font_size(False)
#the_table.set_fontsize(5.5)

npop = 8
nc = 12 
grid = [[0 for x in range(nc)] for y in range(npop)]
def randomgen(high, n):
    listrand = list(np.random.randint(high, size = n))
    return listrand
a = randomgen(6, nc)
a
for i in range (npop):
    grid[i] = randomgen(6,nc)

grid




s = 'individual'
ind = range(1,9)
ind
indname = ["individual" + ' ' + str(i) for i in ind]
indname
data = pandas.DataFrame(grid)
data.index.name = indname
data
checkerboard_table(data)
    plt.show()

def main():
    data = pandas.DataFrame(grid, 
                columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
    checkerboard_table(data,nc,npop)
    

def checkerboard_table(data,nc,npop, fmt='{:.0f}', bkg_colors=['white', 'white']):
    fig, ax = plt.subplots(figsize=(nc*0.8,npop*0.8))
    ax.set_axis_off()
    ax.set_xlabel('Population')
    tb = Table(ax)
    tb.auto_set_font_size(False)
    tb.set_fontsize(14)

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = 0 if val == 1 else 1
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text="individual" + ' ' + str(label + 1), loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.savefig("pop1.pdf",dpi=100)
    return fig

if __name__ == '__main__':
    main()
    

############
    
def checkerboard_table(data,nc,npop,indexname, fmt='{:.0f}', bkg_colors=['yellow', 'white']):
    fig, ax = plt.subplots(figsize=(nc*1.5,npop*1.5))
    ax.set_axis_off()
    ax.set_xlabel('Population')
    tb = Table(ax)
    tb.auto_set_font_size(False)
    tb.set_fontsize(14)

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = 0 if val ==1 else 1
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text= indexname + ' ' + str(label + 1), loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    #plt.savefig("hk.pdf")
    return fig


data = pandas.DataFrame(grid, columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
dataselect = data.loc[[0,5], :]
dataselect
checkerboard_table(dataselect, nc = 12, npop=2)
plt.show()
dataselectcrossover = dataselect.copy()
dataselectcrossover
ind1 = dataselect.iloc[0,:]
ind2 = dataselect.iloc[1,:]
ind2
ind1
ind1[11]
crossind1 = np.copy(ind1)
crossind2 = np.copy(ind2)
crossind1

crossind1 [6:12] = np.copy(ind2[6:12])
crossind1
ind1
crossind2 [6:12] = np.copy(ind1[6:12])
crossind2
ind2
ind1
crossind1
crossind2
datanew = pandas.DataFrame(crossind1,crossind2)
datanew
dataselectcrossover.iloc[0,:] = np.copy(crossind1)
dataselectcrossover.iloc[1,:] = np.copy(crossind2)
dataselectcrossover
dataselect
checkerboard_table(dataselectcrossover, nc = 12, npop=2, indexname = "Individual")

datamutate = dataselectcrossover.copy()
datamutate
datamutate.iloc[0, 10] = 0
datamutate

checkerboard_table(dataselect, nc = 12, npop=2, indexname = "Parent")

checkerboard_table(dataselectcrossover, nc = 12, npop=2, indexname = "Child")

fig.savefig("cross.pdf")

fig.show()

checkerboard_table(dataselect, nc = 12, npop=2,indexname = "Parent")

checkerboard_table(dataselectcrossover, nc = 12, npop=2, indexname = "Child")

checkerboard_table(datamutate, nc = 12, npop=2, indexname = "Child")

dataselectcrossover



def checkerboard_table(data,nc,npop, indexname, fmt='{:.0f}', bkg_colors=['white', 'white']):
    fig, ax = plt.subplots(figsize=(nc*1.5,npop*1.5))
    ax.set_axis_off()
    ax.set_xlabel('Population')
    tb = Table(ax)
    tb.auto_set_font_size(False)
    tb.set_fontsize(20)

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = 0 if val == 1 else 1
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text= indexname + ' ' + str(i+1), loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    #plt.savefig("hk.pdf")
    return fig



checkerboard_table(data= dataselect,nc = 12, npop = 2, indexname = 'Individual')
plt.savefig("parent1.png")
checkerboard_table(data= dataselectcrossover,nc = 12, npop = 2, indexname = 'Offspring')
plt.savefig("crossover.png")
checkerboard_table(data= datamutate,nc = 12, npop = 2, indexname = 'Offspring')
plt.savefig("mutate.png")
plt.show()


def generategraph(file, idx):
    savefile = file + str(idx) + '.png'  # file might need to be replaced by a string
    plt.savefig(savefile)
    plt.show()              # place after plt.savefig()



for idx, fil in enumerate(files):
    generategraph(fil, idx)


###########3
import matplotlib.image as mpimg
img=mpimg.imread('popcopy.png')
img1=mpimg.imread('crossover.png')
img2=mpimg.imread('mutate.png')
imgplot = plt.imshow(img)
plt.show()

fig3 = plt.figure(constrained_layout=True)

gs = fig3.add_gridspec(2, 2)

f3_ax1 = fig3.add_subplot(gs[:, 0])
f3_ax1.set_title('population')
f3_ax1.set_axis_off()
f3_ax2 = fig3.add_subplot(gs[0, 1])
f3_ax2.set_title('crossover')
f3_ax1.set_axis_off()
f3_ax3 = fig3.add_subplot(gs[1, 1])
f3_ax3.set_title('mutation')
f3_ax1.set_axis_off()
f3_ax2.set_axis_off()
f3_ax3.set_axis_off()

f3_ax1.imshow(img)
f3_ax2.imshow(img1)
f3_ax3.imshow(img2)
plt.savefig("all.png", dpi = 700)
plt.show()

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].imshow(img1)
axarr[0].set_title('crossover')
axarr[1].imshow(img2)
axarr[1].set_title('mutation')
axarr[0].set_axis_off()
axarr[1].set_axis_off()
plt.savefig("offspring.png", dpi = 800)
plt.show()

img3=mpimg.imread('offspring.png')
#########

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(img)
ax2.imshow(img3)
ax1.set_axis_off()
ax2.set_axis_off()
plt.savefig("combo.png", dpi = 800)
plt.show()

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)
axarr[0].imshow(img)
axarr[1].imshow(img1)
axarr[1].set_title('crossover')
axarr[2].imshow(img2)
axarr[2].set_title('mutation')
axarr[0].set_axis_off()
axarr[1].set_axis_off()
axarr[2].set_axis_off()
plt.savefig("combo.png", dpi = 800)
plt.show()