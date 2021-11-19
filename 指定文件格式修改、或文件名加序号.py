#功能：列出指定目录下所有py文件(其实文件夹也可以修改），可在文件名前加序号，可统一修改后面的格式
import os
import os.path
path=input('需要更改的文件的路径:')+'/'#"D:/Users/YGG/Desktop/pythonProject/matpltlib/"#指定路径
lst=os.listdir(path)#获取这个路径下的所有文件，包括目录
b=1#文件编号从一开始
for filename in lst:
    if filename.endswith('.txt'):#（可修改）;如果文件以.py结尾,不区分大小写
        # print(filename)
        a = os.path.splitext(filename)  # 分离文件名和扩展名放到元组类型a,('1', '.py')、('2', '.py')。。。，a[0]取出来的是字符类型
        print(type(a))#a是元组，单个元素是字符类型
        print(a[0])  # 0输出文件名,文件名不修改的前提下可运用到下面的命名
        # print(type(a[0]))#
        # newname =str(b)+a[0] + '.txt'  # 文件加序号，还改格式;将b转换为字符连文件名连上后缀名,文件添加序号是随机的，修改文件的格式,大小写无区别
        newname =a[0] + '.py'#文件名不变，只改格式
        b = b + 1
        os.rename(path + filename, path + newname)  # 命名文件或目录，将path + filename这个路径下的文件名修改成path + newname这个路径下的文件名
        # 参数1要修改的目录名(这个目录下的所有文件），参数2修改后的目录名（修改该目录下文件的新名字）
print('已批量修改！')

####