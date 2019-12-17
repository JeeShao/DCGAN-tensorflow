import os

i=1
for file in os.listdir('../../../images64'):    #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
    # print(file)#去掉空格
    os.rename( os.path.join('../../../images64',file), os.path.join('../../../images64',str(i)+os.path.splitext(file)[-1]))
    print(str(i)+os.path.splitext(file)[-1])
    i+=1
