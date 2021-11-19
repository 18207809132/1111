#批量创建、删除、移动、复制的指定文件
import os
import os.path
import shutil

path=input('需要批量创建、删除、移动、复制的指定文件路径是:')+'/'#"D:/Users/YGG/Desktop/pythonProject/matpltlib/"#指定路径
lst=os.listdir(path)#获取这个路径下的所有文件，包括目录

def text_create(name1,msg):#文件批量创建函数，往文件写入msg变量里的东西
    desktop_path=path#批量新建的文件放在哪里，默认为当前源文件夹(可修改放桌面，桌面路径'D:\\Users\\YGG\\Desktop\\'）
    full_path=desktop_path+name1+'.xls'#批量创建xls（如txt创建同名xls),要找到指定路径下有的这个名字的excel文档，没有就创建，并往里面写东西
    file=open(full_path,'w')
    file.write(msg)

# dst_path = input('移动或复制（删除和批量创建功能要注释这一行）到哪个文件夹，该文件夹的路径是:')+'/'#在这里/跟\\都一样
print('________________________________________________________________________\n具体文件是：')
for filename in lst:
    if filename.endswith('.xlsx'):#（可修改）;如果文件以.py结尾(不区分大小写)找出所有这种格式的文件
        print(filename)#仅是字符而已,输出文件全称（名及格式）
        a = os.path.splitext(filename)  # 分离文件名和扩展名放到元组类型a,('1', '.py')、('2', '.py')。。。，a[0]取出来的是字符类型0输出文件名,文件名不修改的前提下可运用到下面的命名# print(a[0])  #

        # text_create(a[0],'测量次序')#D:\Users\YGG\Desktop\文件批量创建（文件批量创建时将下2、3、4行注释）1
        os.remove(path+filename)#文件批量删除,指定路径下的+具体文件（文件批量删除时将134行注释）2
        # shutil.move(path+filename,dst_path)#文件批量移动 3
        # shutil.copy(path+filename,dst_path)#指定文件批量复制 4
print('________________________________________________________________________\n','完成操作!')

