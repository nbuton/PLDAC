file = open("resultats/resultat_finals.txt")
text = ''.join(file.readlines())

cut = text.split("test_score")
tab=[]
for c in cut[1:]:
	cut2 = c.split("[")[1]
	cut2 = cut2.split("]")[0]

	cut3 = cut2.split(",")
	somme=0
	for q in cut3 :
		somme+=float(q)
	#print(somme/5)
	tab.append(somme/5)
print(tab)
print(sum(tab)/14)
