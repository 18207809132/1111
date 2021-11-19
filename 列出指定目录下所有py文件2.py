#walk方法，不仅可以获取指定目录下的文件，而且子目录下的文件也能获取出来
import os
path=os.getcwd()
lst_file=os.walk(path)
# print(lst_file)#这是一个迭代器对象
#遍历迭代器对象，迭代器对象返回来的是一个元组，元组包含目录的路径，目录下所有文件夹和所有文件
for dirpath,dirname,filename in lst_file:
    # print(dirpath)
    # print(dirname)
    # print(filename)
    #  print('____________________________')
    for dir in dirname:
        print(os.path.join(dirpath,dir))#将dirpath和dir联系起来
    print('__________________________________________________________________')
#在当前目录下有多少个子目录
    for file in filename:
        print(os.path.join(dirpath,file))#路径和文件名拼接
    print('__________________________________________________________________')







    # 列出指定目录下所有py文件
    import os
    import os.path

    path = os.getcwd()  # 获取当前目录
    lst = os.listdir(path)  # 获取这个路径下的所有文件，包括目录
    for filename in lst:
        if filename.endswith('.py'):  # 如果文件以.py结尾
            print(filename)


