
with open('train.txt') as fp:
    for line in fp:
	line=line.rstrip('\n');
	words = line.split(";") 
	words[0] = words[0].rstrip('.png')

	filename = words[0]+".txt"
	# Open a file
	fo = open(filename, "w")
	fo.write(words[5]+"\n"+words[1]+" "+words[2]+" "+words[3]+" "+words[4]);

	# Close opend file
	fo.close()
	
