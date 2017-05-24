# Perceptron-binary-classifier-
Perceptron supervised learning model for binary classification of HAM and spam email classification

Instructions to execute PART-2 : Perceptron .
1. make 						       (This will compile the program)
2. java ProgPerceptron {TRAIN_HAM_DIR} {TRAIN_SPAM_DIR} {TEST_HAM_DIR} {TEST_SPAM_DIR} {stop_words.txt} {LEARNING_RATE} {ITERATIONS}	(This will generate both the inorder and out-of-order output_file)
3. make clean 						(Optional : This will clean compiled .class files)

Example:
java ProgPerceptron train/ham/ train/spam test/ham/ test/spam/ stopWords.txt 0.05 100

Output:
Prints the accuracies using the stop words and without using stop words.
