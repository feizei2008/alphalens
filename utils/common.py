import os


def get_file_list(dir, file_list):
    # https://blog.csdn.net/C_chuxin/article/details/83446602
    if file_list is None:
        file_list = []
    new_dir = dir
    if os.path.isfile(dir):
        file_list.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            new_dir = os.path.join(dir, s)
            get_file_list(new_dir, file_list)
    return file_list


