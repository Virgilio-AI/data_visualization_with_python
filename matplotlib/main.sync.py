# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # matplotlib

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

# %%
# %matplotlib inline

# %%
x = np.linspace(1,20,20)
x

# %%
y = randint(1,50,20)

# %%
y

# %%
x.size,y.size

# %%
help(plt)

# %%
plt.plot(y)

# %%
y = y.cumsum()

# %%
plt.plot(y)

# %%
plt.plot(x,y)

# %%
plt.plot(y,color = 'red',marker = "o")

# %%
plt.plot(y,color = 'green',marker = "o",linestyle = '--')

# %% [markdown]
# # Label

# %%
plt.plot(x,y,color = 'r',marker = "o",linestyle = '--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('this is a demo plot')
plt.show()

# %%

# %% [markdown]
# # scatter,bar,hist, and box plots

# %%
x,y

# %%
plt.scatter(x,y,linewidth = 4)
plt.show()

# %%
b = [10,23,43,5,66]
a = ['a','b','c','d','e']
plt.bar(a,b,width = 0.5)
plt.show()

# %%
plt.hist(y,rwidth = 0.8,bins = 30)

# %% [markdown]
# # box plot

# %%
data = [np.random.normal(0,std,2) for std in range(1,4) ]
data.append([0,1,1,2,3,4])

print(data)
# %%
plt.boxplot(data, patch_artist = True)
plt.show()

# %% [markdown]
# # subplot

# %%
x

# %%
y

# %%
y2 = y ** 2

# %%
plt.plot(x,y)
plt.plot(x,y2)

# %%
fig,ax = plt.subplots(2,2)
ax[0,0].plot(x,y)
ax[0,1].plot(x,y2)
ax[1,1].plot(x,y2)
plt.show()
# %%
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.1,0.6,0.3,0.3])

ax1.plot(x,y,'r')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Y plot')

ax2.plot(x,y2,'g')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Y2 plot')

# %%
help(fig.add_axes)

# %% [markdown]
# # xlim, ylim, xticks and yticks

# %%
fig , ax = plt.subplots(1,2,figsize = (12,4))
ax[0].plot(x,y,x,y2)
ax[0].set_xticks([1,4,9])


ax[1].plot(x,y ** 2 ,'black')
ax[1].set_xticks([1,4,9])
ax[1].set_xticklabels([r'$\alpha$',r'$\beta$',r'$\gamma$'])

# %% [markdown]
# # pie plot

# %%
labels = [ 'Frogs','Hogs','Dogs','Lions']
sizes = [15,30,45,10]

# %%
fig, ax = plt.subplots(figsize = (6,6),dpi = 100)
explode = (0.1,0,0,0)
ax.pie(sizes,labels = labels, autopct = '%1.1f%%',shadow = True, startangle = 90,explode = explode)
plt.show()

# %%

# %% [markdown]
# # pie plot text color

# %%
fig, ax = plt.subplots(figsize = (6,6),dpi = 100)
explode = (0.1,0,0,0)
patches,text,autotext = ax.pie(sizes,labels = labels, autopct = '%1.1f%%',shadow = True, startangle = 90,explode = explode)
plt.setp(autotext,size = 14, color = 'blue')
autotext[0].set_color('white')
plt.show()

# %% [markdown]
# # nested pie chart

# %%
fig , ax = plt.subplots(dpi = 100)
size = 0.3

vals = np.array([[60,32],[35,20],[26,36]])
vals_sum = vals.sum(axis = 1)
vals_sum
vals_flat = vals.flatten()

# get the color mappings
cmap = plt.get_cmap("tab20c")
outer_color = cmap(np.arange(3)*8)

inner_color = cmap(np.arange(3)*12)


ax.pie(vals_sum,radius = 1, colors = outer_color, wedgeprops = dict(width = size,edgecolor = 'w'),autopct = '%1.1f%%')


ax.pie(vals_flat,radius = 1 - size, colors = inner_color, wedgeprops = dict(width = size,edgecolor = 'w'),autopct = '%1.1f%%')

plt.show()
print(inner_color)
print(outer_color)

# %%
help(vals.sum)

# %%
help(ax.pie)

# %% [markdown]
# # labeling a pie chart

# %%
fig, ax = plt.subplots(dpi = 100)
recipe = ['375 g flour','75 g sugar','250 g butter','300 g berries']

data = [ float(x.split(" ")[0]) for x in recipe ]
ingredients = [ x.split(" ")[-1] for x in recipe ]


def func(pct, vals):
	absolute = int(pct / 100. * np.sum(vals))

	return "{:.1f}%\n({:d} g)".format(pct, absolute)




wedges, text, autotext = ax.pie(data, autopct = lambda pct: func(pct, data), textprops = dict(color = 'w') )


ax.legend(wedges, ingredients, title = "Ingredients" , loc = "center left", bbox_to_anchor = (1,0, 1,1))

plt.show()

# %% [markdown]
# # Polar axis

# %%
np.random.seed(0)
N = 20
theta = np.linspace(0.0, np.pi * 2 , N , endpoint = False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
colors = plt.cm.plasma(radii / 10)

plt.figure(dpi = 100)
ax = plt.subplot(1,1,1, projection = 'polar')
ax.bar(theta,radii,width = width, bottom = 0, color = colors, alpha = 0.7)
plt.show()

# %%
print("width: ",width)
print("theta: ",theta)
print("radii: ",radii)
print("colors: ",colors)

# %% [markdown]
# # Line plot on a polar axis

# %%
r = np.arange(0,2,0.01)
theta = 2 * np.pi * r
plt.figure(dpi = 100)
ax = plt.subplot(111, projection = 'polar')
ax.plot(theta,r)
ax.set_rmax(2)
ax.set_rticks([0.0,0.5,1.0,1.5,2.0])
ax.set_rlabel_position(-30)
ax.grid(True)
plt.show()

# %% [markdown]
# # scatter plot on a polar axis

# %%
np.random.seed(0)

N = 150
theta = 2 * np.random.rand(N)
area = 50 * r * np.pi * r ** 2
colors = theta

# %% [markdown]
# # scatter plot on a polar axis

# %%
np.random.seed(0)

N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 50 * 2 * np.pi * r**2
colors = theta

plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection = 'polar')
c = ax.scatter(theta,r,c = colors, s = area, cmap = 'hsv',alpha = 0.8)
plt.show()

# %% [markdown]
# # Integral as the area under a curve

# %%
from matplotlib.patches import Polygon


# %%
def func(x):
	return (x - 3) * ( x - 5) * ( x - 7 ) + 85

a,b = 2,9
x = np.linspace(0,10)
y = func(x)

fig, ax = plt.subplots(dpi = 100)
ax.plot(x,y,'r',linewidth = 2)

ax.set_ylim(bottom = 0)

ix = np.linspace(a,b)
iy = func(ix)
verts = [(a,0),*zip(ix,iy), (b,0) ]
poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
ax.add_patch(poly)

ax.text(0.5 * (a + b) , 30 , r'$\int_a^b{f(x)dx}$')

fig.text(0.9,0.05,'x')
fig.text(0.1,0.9,'y')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

ax.set_xticks((a,b))
ax.set_xticklabels(('$a$','$b$'))
ax.set_yticks([])
plt.show()

# %% [markdown]
# # Real-Time cpu uses plot

# %%
from matplotlib.animation import FuncAnimation

# %%
from psutil import cpu_percent

# %%
cpu_percent()

# %%
uses = []

# %%
for i in range(20):
	cpu_use = cpu_percent()
	uses.append(cpu_use)

# %%
print(cpu_use)

# %%
plt.plot(uses)

# %%
# %matplotlib notebook
frame_len = 200
y = []

fig = plt.figure(dpi = 100)
ax = plt.figure(figsize = (8,6))
def animate(i):
	y.append(cpu_percent())
	if len(y) <= frame_len:
		plt.cla()
		plt.plot(y, 'r', label = 'Real TIme cpu use')
	else:
		plt.cla()
		plt.plot(y[-frame_len:],'r', label = 'Real TIme cpu use')
	plt.ylim(0,100)
	plt.xlabel('Time (s)')
	plt.ylabel('CPU uses (%)')
	plt.legend(loc = 'upper right')
	plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval = 500)
