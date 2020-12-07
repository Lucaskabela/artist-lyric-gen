

with open("persona_tags_bpe.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
maxlen = 0
content = [x.strip() for x in content] 
maxstring = ""
for mystr in content:
	if "A" in mystr: 
		if "Y" in mystr:
			newstr = (mystr[:mystr.index("A")] + mystr[mystr.index("Y"):])
			maxlen = max(len(newstr.split()), maxlen)
			if maxlen == len(newstr.split()):
				maxstring = newstr
			print(newstr)
		else:
			newstr = (mystr[:mystr.index("A")] )
			maxlen = max(len(newstr.split()), maxlen)
			if maxlen == len(newstr.split()):
				maxstring = newstr
			print(newstr)
	else:
		maxlen = max(len(mystr), maxlen)
		if maxlen == len(mystr):
				maxstring = newstr
print(maxlen)
print(maxstring)