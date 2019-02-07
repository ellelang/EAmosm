import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator	
# --- Create grid of numbers
# Create an empty list
grid = []
# Loop for each row
for row in range(10):
    # For each row, create a list that will
    # represent an entire row
    grid.append([])
    # Loop for each column
    for column in range(10):
        # Add a the number zero to the current row
        grid[row].append(0)
        
grid
grid
N = 10
fig, ax = plt.subplots(1, 1, tight_layout=True)

for (i, j), z in np.ndenumerate(grid):
    ax.text(j, i, '{:0.1f}'.format(z),
            verticalalignment='bottom',
            horizontalalignment='left')
#for x in range(N+1):
#    ax.axhline(x, lw=2, color='k', zorder=4)
#    ax.axvline(x, lw=2, color='k', zorder=4)
    
# turn off the axis labels
ax.imshow(grid, cmap="binary",extent=[0, N, 0, N],zorder=0)
ax.axis('off')
ax.grid(which = 'minor')

grid




spacing = 1
minorLocator = MultipleLocator(spacing)
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
fig, ax = plt.subplots(1, 1, tight_layout=True)
for (i, j), z in np.ndenumerate(grid):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.yaxis.set_minor_locator(minorLocator)
plt.xaxis.set_minor_locator(minorLocator)

plt.grid(which = 'minor')
plt.show()

for (i, j), z in np.ndenumerate(grid):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


N = 10
# make an empty data set
data = np.ones((N, N)) * np.nan
# fill in some fake data
for j in range(3)[::-1]:
    data[N//2 - j : N//2 + j +1, N//2 - j : N//2 + j +1] = j
# make a figure + axes
fig, ax = plt.subplots(1, 1, tight_layout=True)
# make color map
my_cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
# set the 'bad' values (nan) to be white and transparent
my_cmap.set_bad(color='w', alpha=0)
# draw the grid
for x in range(N + 1):
    ax.axhline(x, lw=2, color='k', zorder=5)
    ax.axvline(x, lw=2, color='k', zorder=5)
# draw the boxes
ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)
# turn off the axis labels
ax.axis('off')

fig, ax = plt.subplots(1, 1, tight_layout=True)

spacing = 1
minorLocator = MultipleLocator(spacing)
plt.plot(9 * np.random.rand(10))
# Set minor tick locations.
plt.yaxis.set_minor_locator(minorLocator)
plt.xaxis.set_minor_locator(minorLocator)
# Set grid to use minor tick locations. 
plt.grid(which = 'minor')


savefig('figname.png', facecolor=fig.get_facecolor(), transparent=True)


import matplotlib.pyplot as plt
import numpy as np
import pandas

from matplotlib.table import Table

data = pandas.DataFrame(grid, 
columns=['1','2','3','4','5','6','7','8','9','10'])
data
plt.show
tb = Table(ax)	
def main():
    data = pandas.DataFrame(grid, 
                columns=['1','2','3','4','5','6','7','8','9','10'])
    checkerboard_table(data)
    plt.show()

def checkerboard_table(data, fmt='{:.1f}', bkg_colors=['white', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax,bbox=[0,0,1,1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j +1) % 2][(i) % 2]
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width/2, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig

if __name__ == '__main__':
    main()
    
    
import matplotlib.pyplot as plt
import numpy as np

data = np.random.random((4, 4))

fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(grid, cmap='binary')

for (i, j), z in np.ndenumerate(grid):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
ax.grid(which='minor', linestyle='-', linewidth='0.5', color='red')
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Make a 9x9 grid...
nrows, ncols = 9,9
image = np.zeros(nrows*ncols)

# Set every other cell to a random number (this would be your data)
image[::2] = np.random.random(nrows*ncols //2 + 1)

# Reshape things into a 9x9 grid.
image = image.reshape((nrows, ncols))
for i, (image_row, data_row) in enumerate(zip(image, data)):
    image_row[i%2::2] = data_row
row_labels = range(nrows)
col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
plt.matshow(image,cmap = 'binary')
plt.xticks(range(ncols), col_labels)
plt.yticks(range(nrows), row_labels)
plt.show()