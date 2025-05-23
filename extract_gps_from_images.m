image_folder = './benchmark/2k'; % 修改为你的图像路径
output_file = 'gps_groundtruth.json';

images = dir(fullfile(image_folder, '*.jpg'));
results = struct();

for i = 1:length(images)
    img_path = fullfile(images(i).folder, images(i).name);
    info = imfinfo(img_path);
    if isfield(info, 'Comment')
        comment = info.Comment;
        if iscell(comment)
            comment = comment{1}; % 提取第一个注释
        end
        lat_expr = 'GPSLatitude=([-+]?\d+(\.\d+)?)';
        lon_expr = 'GPSLongitude=([-+]?\d+(\.\d+)?)';

        lat_tokens = regexp(comment, lat_expr, 'tokens');
        lon_tokens = regexp(comment, lon_expr, 'tokens');

        if ~isempty(lat_tokens) && ~isempty(lon_tokens)
            lat = str2double(lat_tokens{1}{1});
            lon = str2double(lon_tokens{1}{1});
            results.(images(i).name) = struct('lat', lat, 'lon', lon);
        end
    end
end

json_text = jsonencode(results);
fid = fopen(output_file, 'w');
fwrite(fid, json_text, 'char');
fclose(fid);
