rule all:
	input:	
		expand('DataDNN.txt', x=range(0,2,1))
		
#Data creation for attacker and user: same parameters and size
rule A1:
	input:
		"main.py"
	output:
		file1="Authen_{x}.txt"
	shell:
		'python main.py shots1 dist1000 wait1000 users1 cores1 attack{wildcards.x}'

#Run the binary classification with the data Authen_0.txt and Authen_1.txt
rule A2:
	input:
		"DNN_binary_class.py",
		"Authen_0.txt",
		"Authen_1.txt"
		#expand("Authen_{x}.txt",x=range(0,2,1))
	output:
		"DataDNN.txt"
	shell:
		'python DNN_binary_class.py'