# https://medium.com/@nurheliza_dede/mapping-geoscience-with-matplotlib-python-75661d6d0661
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
#%%
fig = plt.figure(figsize=(20,10))
map = Basemap(projection='merc',llcrnrlon= 122.22, llcrnrlat=-11.37,urcrnrlon=123.97, urcrnrlat=-10.12, resolution='i')
map.drawcoastlines(linewidth=2)
map.drawcountries(linewidth=2)
map.drawstates(linewidth=2)
map.drawmapboundary(fill_color='#99ffff')
map.drawparallels(np.arange(-11.37,-10.12, 0.3),labels=[1,0,0,0],fontsize=20)
map.drawmeridians(np.arange(122.22,123.97,0.3),labels=[0,0,0,1],fontsize=20)

x, y = map(lons, lats)
map.contourf(x,y, windspeed, cmap=get_cmap('Blues'),extend='both',zorder=1)
contourf = map.colorbar(
    map.contourf(x, y, windspeed, cmap=get_cmap('Blues'),extend='both',zorder=1),
    location='bottom', size='4.7%', pad='17.6%'
)
vecplot = map.quiver(x, y, u, v, scale=100, width = 0.005, headwidth = 3, headlength = 5)
#%%
# https://basemaptutorial.readthedocs.io/en/latest/plotting_data.html
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


map = Basemap(projection='ortho',
              lat_0=0, lon_0=0)

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()


x, y = map(2, 41)
x2, y2 = (-90, 10)

plt.annotate('Barcelona', xy=(x, y),  xycoords='data',
                xytext=(x2, y2), textcoords='offset points',
                color='r',
                arrowprops=dict(arrowstyle="fancy", color='g')
                )

x2, y2 = map(0, 0)
plt.annotate('Barcelona', xy=(x, y),  xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->")
                )
plt.show()
#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np


map = Basemap(llcrnrlon=-93.7, llcrnrlat=28., urcrnrlon=-66.1, urcrnrlat=39.5,
              projection='lcc', lat_1=30., lat_2=60., lat_0=34.83158, lon_0=-98.)

ds = gdal.Open("../sample_files/wrf.tiff")
lons = ds.GetRasterBand(4).ReadAsArray()
lats = ds.GetRasterBand(5).ReadAsArray()
u10 = ds.GetRasterBand(1).ReadAsArray()
v10 = ds.GetRasterBand(2).ReadAsArray()

x, y = map(lons, lats)

yy = np.arange(0, y.shape[0], 4)
xx = np.arange(0, x.shape[1], 4)

points = np.meshgrid(yy, xx)

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
map.drawcoastlines(color = '0.15')

map.barbs(x[points], y[points], u10[points], v10[points],
    pivot='middle', barbcolor='#333333')


plt.show()