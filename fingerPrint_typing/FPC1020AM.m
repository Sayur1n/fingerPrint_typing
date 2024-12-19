pe = pyenv();
disp(pe);

% 可以手动设置 Python 解释器的路径
% 注意1：最好使用全局python解释器，而不是conda包管理环境中的版本（容易出错）
% 参考 https://ww2.mathworks.cn/help/matlab/matlab_external/install-supported-python-implementation.html
% pyenv('Version', 'path_to_your_python/python.exe');
% 注意2：请将本代码文件和python文件放置在同一目录下

% 确保 Python 路径中包含所需的模块
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% 导入模块
sensor_module = py.importlib.import_module('driver_fpc1020am'); %导入fp_python文件
py.importlib.reload(sensor_module);  % 如果模块已在 Python 会话中导入过，可使用 reload


% 创建 Sensor 对象，并调用初始化方法
sensor = sensor_module.DriverFPC1020AM();
figure;
while true
    img = sensor.get_image();
    if isa(img, 'py.numpy.ndarray')
        img_matlab = double(py.array.array('d', py.numpy.nditer(img))); % 将数据处理为列向量
        img_matlab = reshape(img_matlab, [192,192]); % 根据尺寸重构为原来的样子
        img_matlab = transpose(uint8(img_matlab)); % 转换为 uint8      
        imshow(img_matlab);
        pause(0.1);
    else
        warning('Skip the Frame');
        % break;
    end
end
