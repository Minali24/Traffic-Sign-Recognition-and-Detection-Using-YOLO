import sys
import os

with open(sys.argv[1]) as f:
    lines = f.readlines()
    f.close()

max_width = 1360
max_height = 800

lines = [l.strip('\n') for l in lines]
out_folder = sys.argv[2]

filelist = [ f for f in os.listdir(out_folder) ]
for f in filelist:
    os.remove(out_folder + f)

for l in lines:
	l = l.split(";")
	name = l[0].split('.')[0]
	left_col = int(l[1])
	up_row = int(l[2])
	right_col = int(l[3])
	down_row = int(l[4])
	class_no = int(l[5])

	box_width = float(right_col - left_col)
	box_height = float(down_row - up_row)

	op = str(class_no) + " " + str(left_col/max_width) + " " + str(up_row/max_height) + " " + str(box_width/max_width) + " " + str(box_height/max_height)

	new_file_name = out_folder + name + ".txt"

	if os.path.isfile(new_file_name):
		with open(new_file_name, 'a') as file:
			file.write("\n" + op)
	else:
		with open(new_file_name, 'w') as file:
			file.write(op)
	file.close()
