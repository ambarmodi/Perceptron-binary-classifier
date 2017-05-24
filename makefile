all: HW3

HW3: PerceptronModel.java ProgPerceptron.java
	javac *.java

clean:
	rm -rf *.class
