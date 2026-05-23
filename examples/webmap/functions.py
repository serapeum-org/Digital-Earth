import os, sys
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
# from functions import CreatePoint, PrepareData
import folium
from folium import plugins
from folium.features import GeoJsonPopup, GeoJsonTooltip
import branca.colormap as cm

def CreatePoint(coords):
    """CreatePoint

    CreatePoint takes a list of tuples of coordinates and convert it into
    a list of Shapely point object

    inputs
    ----------
        1-coords : [List]
            list of tuples [(x1,y1),(x2,y2)] or [(long1,lat1),(long2,lat1)]

    Outputs:
    ----------
        1-points : [List]
            list of Shaply point objects [Point,Point]

    Examples:
    ----------
        coordinates = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
        PointList = GIS.CreatePoint(coordinates)
        # to assign these objects to a geopandas dataframe
        # NopreviousGeoms is the number of geometries already exists in the
        # geopandas dataframe
        NopreviousGeoms = 5
        for i in range(NopreviousGeoms,NopreviousGeoms+len(PointList)):
            NewGeometry.loc[i,'geometry'] = PointList[i-NopreviousGeoms]
    """
    points = list()
    for i in range(len(coords)):
        points.append(Point(coords[i]))

    return points

def PrepareData(path):
    # rootpath = os.path.abspath(os.getcwd())
    # sys.path.append(rootpath + "/src")
    # datapath = os.path.join(rootpath, "../data")
    # print(datapath)
    # os.chdir(rootpath)

    #  read the data
    print("- Start reading the data")
    data = gpd.read_file(path + "/RaCa_general_location.csv")
    # print(data.shape)
    data2 = gpd.read_file(path + "/RaCA_SOC_pedons.csv")
    # print(data2.shape)
    # unify column name for merging
    data = data.rename(columns={"RaCA_Id": "rcasiteid"})
    #  check duplicate
    # print(data.shape)
    data.drop_duplicates(["rcasiteid"], inplace=True)
    # print(data.shape)
    print("- Check duplicate data")
    # print(data2.shape)
    data2.drop_duplicates(["rcasiteid"], inplace=True)
    # print(data2.shape)
    #  merge
    new_data = data2.merge(data, on='rcasiteid', how='inner', sort=False)
    # convert to  geodataframe
    new_data = gpd.GeoDataFrame(new_data)
    # print(new_data.shape)
    # new_data.drop_duplicates(["rcasiteid"], inplace=True)
    # print(new_data.shape)
    #  create a shapely geometry object for the geopanda table
    # create the lat and long as a list of tuples
    lats = new_data.loc[:, 'Gen_lat'].tolist()
    lons = new_data.loc[:, 'Gen_long'].tolist()

    lats = [float(i) for i in lats]
    lons = [float(i) for i in lons]

    # for i in range(len())
    coords = list(zip(lons, lats))
    # not necessary
    print("- Create Shapely point object")
    objects = CreatePoint(coords)
    new_data['geometry'] = objects
    del new_data['geometry_y'], new_data['geometry_x']

    return new_data
    # new_data.to_file("data/data.geojson", driver="GeoJSON")
    print("- Preparing the data has successfully finished")



def CreateMap(path):
    # os.chdir("F:/01Algorithms/webmap")
    # rootpath = os.path.abspath(os.getcwd())
    # sys.path.append(rootpath + "/src")
    # datapath = os.path.join(rootpath, "data")
    # os.chdir(rootpath)

    data = PrepareData(path)
    # data = gpd.read_file("data/data.geojson", driver="GeoJSON")
    print("- Start creating the map")
    for i in range(len(data)):
        try:
            data.loc[i, 'SOCstock30'] = float(data.loc[i, 'SOCstock30'])
        except:
            data.loc[i, 'SOCstock30'] = -1  # np.nan
            # print(str(i) + "SOCstock30 is NA")
        try:
            data.loc[i, 'SOCstock5'] = float(data.loc[i, 'SOCstock5'])
        except:
            data.loc[i, 'SOCstock5'] = -1  # np.nan
            # print(str(i) + "SOCstock5 is NA")

        try:
            data.loc[i, 'SOCstock100'] = float(data.loc[i, 'SOCstock100'])
        except:
            data.loc[i, 'SOCstock100'] = -1  # np.nan
            # print(str(i) + "SOCstock10 is NA")
    # prepare the color and the size
    # data['color30'] = data['SOCstock30'].apply(lambda SOCstock30:"Black" if SOCstock30>=400 else
    #                                          "green" if SOCstock30>=300 and SOCstock30<400 else
    #                                          "Orange" if SOCstock30>=200 and SOCstock30<300 else
    #                                          "darkblue" if SOCstock30>=150 and SOCstock30<200 else
    #                                          "red" if SOCstock30>=100 and SOCstock30<150 else
    #                                          "lightblue" if SOCstock30>=75 and SOCstock30<100 else
    #                                          "brown" if SOCstock30>=50 and SOCstock30<75 else
    #                                          "grey")
    print("- Calculate Size of Points")
    data['size30'] = data['SOCstock30'].apply(lambda SOCstock30: 12 if SOCstock30 >= 400 else
                                                10 if SOCstock30 >= 300 and SOCstock30 < 400 else
                                                8 if SOCstock30 >= 200 and SOCstock30 < 300 else
                                                6 if SOCstock30 >= 150 and SOCstock30 < 200 else
                                                4 if SOCstock30 >= 100 and SOCstock30 < 150 else
                                                2 if SOCstock30 >= 75 and SOCstock30 < 100 else
                                                1 if SOCstock30 >= 50 and SOCstock30 < 75 else
                                                0.1)

    data['size100'] = data['SOCstock100'].apply(lambda SOCstock100: 12 if SOCstock100 >= 400 else
                                                10 if SOCstock100 >= 300 and SOCstock100 < 400 else
                                                8 if SOCstock100 >= 200 and SOCstock100 < 300 else
                                                6 if SOCstock100 >= 150 and SOCstock100 < 200 else
                                                4 if SOCstock100 >= 100 and SOCstock100 < 150 else
                                                2 if SOCstock100 >= 75 and SOCstock100 < 100 else
                                                1 if SOCstock100 >= 50 and SOCstock100 < 75 else
                                                0.1)

    data['size5'] = data['SOCstock5'].apply(lambda SOCstock5: 12 if SOCstock5 >= 400 else
                                            10 if SOCstock5 >= 300 and SOCstock5 < 400 else
                                            8 if SOCstock5 >= 200 and SOCstock5 < 300 else
                                            6 if SOCstock5 >= 150 and SOCstock5 < 200 else
                                            4 if SOCstock5 >= 100 and SOCstock5 < 150 else
                                            2 if SOCstock5 >= 75 and SOCstock5 < 100 else
                                            1 if SOCstock5 >= 50 and SOCstock5 < 75 else
                                            0.1)

    colormap = cm.StepColormap(
        ["black", "lightblue", "brown","darkblue", "Orange", "green","#DC143C",],
        index=[50, 75, 100, 150, 200, 300, 400], caption="SOCstock",
        vmin=50, vmax=400,
    )

    # colormap = cm.linear.RdYlBu_05.to_step(data = data['SOCstock30'], n = 8, method = 'quantiles')

    datajson = data.to_json()
    datadict = data.to_dict()

    popup = GeoJsonPopup(
        fields=["rcasiteid", "SOCstock30", "SOCstock5", "SOCstock100"],
        aliases=["Site ID", "SOCstock30", "SOCstock5", "SOCstock100"],
        localize=True,
        labels=True,
        style="background-color: yellow;",
    )

    tooltip = GeoJsonTooltip(
        fields=["rcasiteid", "SOCstock30", "SOCstock5", "SOCstock100"],
        aliases=["Site ID", "SOCstock30", "SOCstock5", "SOCstock100"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )

    def color30(feature):
        # print("color30")
        # return datadict['color30'][int(feature['id'])]
        return colormap(feature['properties']['SOCstock30'])

    def color100(feature):
        # print("color100")
        # print(colormap(feature['properties']['SOCstock100']))
        # return datadict['color30'][int(feature['id'])]
        return colormap(feature['properties']['SOCstock100'])

    def color5(feature):
        # print("color5")
        # return datadict['color30'][int(feature['id'])]
        return colormap(feature['properties']['SOCstock5'])

    def size30(feature):
        return datadict['size30'][int(feature['id'])]

    def size100(feature):
        return datadict['size100'][int(feature['id'])]

    def size5(feature):
        return datadict['size5'][int(feature['id'])]

    map = folium.Map([40, -100], zoom_start=5)
    # names = ["SOCstock30 - Markers", "SOCstock100 - Markers", "SOCstock5 - Markers"]
    # colorfn = [color30, color100, color5]
    # sizefn = [size30, size100, size5 ]
    # for i in range(3):
    #     folium.GeoJson(
    #         datajson,
    #         name=names[i],
    #         marker=folium.CircleMarker(fill_color='orange', radius=4,
    #                                    fill_opacity=0.7, color="black", weight=1),
    #         popup = popup,
    #         tooltip = tooltip,
    #         style_function = lambda feature: {
    #             "fillColor": colorfn[i](feature),
    #             "radius": sizefn[i](feature),
    #         },
    #         highlight_function=lambda x: {"fillOpacity": 0.8},
    #         zoom_on_click=True,
    #     ).add_to(map)
    #     # folium.LayerControl().add_to(map)
    print("- Create the first Map - SOCstock 100")
    folium.GeoJson(
        datajson,
        name="SOCstock100 - Markers",
        marker=folium.CircleMarker(fill_color='orange', radius=4,
                                   fill_opacity=0.7, color="black", weight=1),
        popup=popup,
        tooltip=tooltip,
        style_function=lambda feature: {
            "fillColor": color100(feature),
            "radius": size100(feature),
        },
        highlight_function=lambda x: {"fillOpacity": 0.8},
        zoom_on_click=True,
    ).add_to(map)
    print("- Create the second Map - SOCstock 30")
    folium.GeoJson(
        datajson,
        name="SOCstock30 - Markers",
        marker=folium.CircleMarker(fill_color='orange', radius=4,
                                   fill_opacity=0.7, color="black", weight=1),
        # popup = popup,
        # tooltip = tooltip,
        style_function=lambda feature: {
            "fillColor": color30(feature),
            "radius": size30(feature),
        },
        highlight_function=lambda x: {"fillOpacity": 0.8},
        zoom_on_click=True,
    ).add_to(map)

    map.add_child(colormap)
    print("- Create the third Map - SOCstock 5")
    folium.GeoJson(
        datajson,
        name="SOCstock5 - Markers",
        marker=folium.CircleMarker(fill_color='orange', radius=4,
                                   fill_opacity=0.7, color="black", weight=1),
        # popup = popup,
        # tooltip = tooltip,
        style_function=lambda feature: {
            "fillColor": color5(feature),
            "radius": size5(feature),
        },
        highlight_function=lambda x: {"fillOpacity": 0.8},
        zoom_on_click=True,
    ).add_to(map)

    map.add_child(folium.LatLngPopup())
    # map.save('RCS.html')
    print("- Create Heat Maps")
    # heat map
    location_data = data.loc[:, ['Gen_lat', 'Gen_long', 'size100']].values
    map.add_child(plugins.HeatMap(location_data, radius=10, name="SOCstock100 - Heat Map ", ))
    location_data = data.loc[:, ['Gen_lat', 'Gen_long', 'size30']].values
    map.add_child(plugins.HeatMap(location_data, radius=10, name="SOCstock30 - Heat Map ", ))
    location_data = data.loc[:, ['Gen_lat', 'Gen_long', 'size5']].values
    map.add_child(plugins.HeatMap(location_data, radius=10, name="SOCstock5 - Heat Map ", ))
    # map.save('RCS.html')
    print("- Create different tiles")
    folium.TileLayer('cartodbpositron').add_to(map)
    folium.TileLayer('cartodbdark_matter').add_to(map)
    folium.TileLayer('Stamen Terrain').add_to(map)
    folium.TileLayer('Stamen Toner').add_to(map)
    folium.TileLayer('Stamen Water Color').add_to(map)

    print("- Create Layer control")
    folium.LayerControl().add_to(map)
    print("- Save map as html")
    map.save('src/RCS.html')
    print("- Map is created successfully")