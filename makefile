all: HW3

HW3: LogisticRegression.java PerceptronModel.java ProgPerceptron.java ProgramLR.java
	javac *.java

clean:
	rm -rf *.class