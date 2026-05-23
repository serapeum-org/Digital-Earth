# https://contextily.readthedocs.io/en/latest/intro_guide.html
import matplotlib
matplotlib.use("TkAgg")
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
#%%
data_url = "https://ndownloader.figshare.com/files/20232174"
gdf = gpd.read_file(data_url)

#%%
ax = gdf.plot(color="red", figsize=(9, 9))
cx.add_basemap(ax, crs=gdf.crs.to_string())

#%%
zaragoza = gdf.query("city_id == 'ci122'")
ax = zaragoza.plot(facecolor="none",
                   edgecolor="red",
                   linewidth=2
                  )
cx.add_basemap(ax,
               crs=zaragoza.crs.to_string(),
               source=cx.providers.CartoDB.Voyager
              )
#%% Coordinate-based searches¶
west, south, east, north = (
    3.616218566894531,
    50.98912458110244,
    3.8483047485351562,
    51.13994019806845
             )

ghent_img, ghent_ext = cx.bounds2img(
    west, south, east, north, ll=True, source=cx.providers.Stamen.Toner
)
ghent_img.shape
ghent_ext

f, ax = plt.subplots(1, figsize=(9, 9))
ax.imshow(ghent_img, extent=ghent_ext)
#%% Places
nightlights = cx.providers.NASAGIBS.ViirsEarthAtNight2012
ireland = cx.Place("Ireland", source=nightlights)
ireland.plot()
#%% Store basemaps locally
tempe = cx.Place("Tempe, AZ")
tempe.plot()

bristol = cx.Place("Bristol, UK", path="bristol.tif")

w, s, e, n = (-3.0816650390625,
              53.268087670237485,
             -2.7582550048828125,
              53.486002749115556)
_ = cx.bounds2raster(w, s, e, n,
                     ll=True,
                     path="liverpool.tif",
                     source=cx.providers.CartoDB.Positron
                    )