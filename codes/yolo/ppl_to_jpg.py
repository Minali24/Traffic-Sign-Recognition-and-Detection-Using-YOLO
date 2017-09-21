from PIL import Image
from os import walk, makedirs
from os.path import exists

folder = "FullIJCNN2013"
mypath = "/opt/labuser/FullIJCNN2013"
save_path = "/opt/labuser/Desktop/Converted_Images"

files = []
for (dirpath, dirnames, filenames) in walk(mypath):
	files += list(map(lambda x: dirpath+"/"+x,filenames))

files = [f for f in files if f.endswith(".ppm")]

count=1
for i in files:
	im = Image.open(i)
	save_file_name = save_path + '/' + ((i.split('/',4)[-1]).split(".")[0]) + ".jpg"
	
	directory = i.split('/')[-2]
	if directory != folder:
		directory = save_path+"/"+directory
		if not exists(directory):
			print(directory)
			makedirs(directory)

	im.save(save_file_name)
	print("File #",count,":",save_file_name)
	count+=1