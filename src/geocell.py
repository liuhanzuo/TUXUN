import json

# 替换为你的ADM1.geojson文件路径
file_path = 'ADM1.geojson'

with open(file_path, 'r') as file:
    data = json.load(file)

tot = 0
shape_centers = []
for feature in data['features']:
    properties = feature['properties']
    geometry = feature['geometry']
    
    shape_name = properties['shapeName']
    shape_type = geometry['type']
    coordinates = geometry['coordinates']
    
    # print(f"Shape Name: {shape_name}")
    # print(f'Shape Type: {shape_type}')
    cur = {}
    cur['shapeName'] = shape_name
    centers = []
    if shape_type == "MultiPolygon":
        for multipolygon in coordinates:
            for polygon in multipolygon:
                # for point in polygon:
                #     print(f" - {point}")
                latitude_sum = 0.
                longitude_sum = 0.
                for point in polygon:
                    latitude_sum += point[0]
                    longitude_sum += point[1]
                center = [latitude_sum / len(polygon), longitude_sum / len(polygon)]
                centers.append(center)
    else:
        for polygon in coordinates:
            # for point in polygon:
            #     print(f" - {point}")
            latitude_sum = 0.
            longitude_sum = 0.
            for point in polygon:
                latitude_sum += point[0]
                longitude_sum += point[1]
            center = [latitude_sum / len(polygon), longitude_sum / len(polygon)]
            centers.append(center)
    # print(shape_name, centers)
    center_of_centers = [0., 0.]
    for center in centers:
        center_of_centers[1] += center[0]
        center_of_centers[0] += center[1]
    center_of_centers[0] /= len(centers)
    center_of_centers[1] /= len(centers)
    cur['center'] = center_of_centers
    shape_centers.append(cur)
    tot += 1

output_file_path = 'shape_centers.json'
with open(output_file_path, 'w') as outfile:
    json.dump(shape_centers, outfile, indent=4)

print(f"Shape centers have been saved to {output_file_path}")