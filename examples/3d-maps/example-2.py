import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m = Basemap(projection='ortho', lat_0=50, lon_0=-100, resolution='l', ax=ax)


# Sample data
lat = [40, 45, 50]
lon = [-110, -100, -90]
depth = [1000, 2000, 3000]

# Convert lat/lon to x/y using Basemap
x, y = m(lon, lat)

# Plot 3D points
ax.scatter(x, y, depth)

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')
#%%
# Set title
ax.set_title('3D Plot with Basemap')

# ax = Axes3D(fig)

# Set map boundaries
# m.drawcoastlines()
# m.fillcontinents()

ax.add_collection3d(m.drawcoastlines(linewidth=0.25))
ax.add_collection3d(m.drawcountries(linewidth=0.35))

# Add colorbar
cbar = plt.colorbar(ax.scatter(x, y, depth))
cbar.set_label('Depth')

# Save or display the plot
# plt.savefig('3d_plot.png')
plt.show()
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

map = Basemap()

fig = plt.figure()
ax = Axes3D(fig)

'''
ax.azim = 270
ax.elev = 90
ax.dist = 5
'''

ax.add_collection3d(map.drawcoastlines(linewidth=0.25))
ax.add_collection3d(map.drawcountries(linewidth=0.35))

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and an axes object for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a Basemap object and set the map projection parameters
m = Basemap(projection='merc', llcrnrlon=-180, llcrnrlat=-80, urcrnrlon=180, urcrnrlat=80, resolution='l', ax=ax)

# Generate sample data
lons = np.linspace(-180, 180, 100)
lats = np.linspace(-80, 80, 50)
lons, lats = np.meshgrid(lons, lats)
elevation = np.sin(np.radians(lats)) * np.cos(np.radians(lons))

# Convert lat/lon to x/y using Basemap
x, y = m(lons, lats)

# Plot 3D surface
ax.plot_surface(x, y, elevation, cmap='viridis')

# Add labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation')

# Set title
ax.set_title('3D Surface Plot on Map')

# Set map boundaries
m.drawcoastlines()
m.fillcontinents()

ax = Axes3D(fig)
ax.add_collection3d(m.drawcoastlines(linewidth=0.25))

# Save or display the plot
plt.savefig('3d_surface_plot.png')
plt.show()
#%%
import itertools

import cartopy
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig, xlim=[-180, 180], ylim=[-90, 90])
ax.set_zlim(bottom=0)


concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

target_projection = ccrs.PlateCarree()

feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
geoms = feature.geometries()

geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]

paths = concat(geos_to_path(geom) for geom in geoms)

polys = concat(path.to_polygons() for path in paths)

lc = PolyCollection(polys, edgecolor='black',
                    facecolor='green', closed=False)

ax.add_collection3d(lc)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

# Create a figure and an axes object for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a Cartopy map projection
projection = ccrs.Mercator()

# Generate sample data
lons = np.linspace(-180, 180, 100)
lats = np.linspace(-80, 80, 50)
lons, lats = np.meshgrid(lons, lats)
elevation = np.sin(np.radians(lats)) * np.cos(np.radians(lons))

# Convert lat/lon to x/y using the projection
x, y, _ = projection.transform_points(ccrs.PlateCarree(), lons, lats).T

# Plot 3D surface
ax.plot_surface(x, y, elevation, cmap='viridis')

# Add labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation')

# Set title
ax.set_title('3D Surface Plot on Map')

# Set map boundaries
ax.set_xlim3d(-2e7, 2e7)
ax.set_ylim3d(-2e7, 2e7)
ax.set_zlim3d(-1, 1)  # Adjust z-limits according to your data

# Save or display the plot
# plt.savefig('3d_surface_plot.png')
plt.show()
