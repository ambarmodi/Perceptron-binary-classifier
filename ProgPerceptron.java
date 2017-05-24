
public class ProgPerceptron {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		/*String trainingHamDir = "D:\\Courses\\ML\\HW3\\train\\ham";
		String trainingSpamDir = "D:\\Courses\\ML\\HW3\\train\\spam";
		String testingHamDir = "D:\\Courses\\ML\\HW3\\test\\ham";
		String testingSpamDir ="D:\\Courses\\ML\\HW3\\test\\spam";
		String stopWord = "D:\\Courses\\ML\\HW3\\stopWords.txt";*/
		
		double accuracy,accuracySW;
		if(args.length != 7) {
			System.out.println("Invalid Arguments: Please refer the README.txt for instructions");
			return;
		}

		String trainingHamDir = args[0];
		String trainingSpamDir = args[1];
		String testingHamDir = args[2];
		String testingSpamDir = args[3];
		String stopWord = args[4];
		PerceptronModel.eta = Double.parseDouble(args[5]);
		PerceptronModel.iterations = Integer.parseInt(args[6]);
		
		try {	
			System.out.println("===Training Perceptron model started without using stopWords===");
			PerceptronModel perceptron = new PerceptronModel(trainingHamDir, trainingSpamDir);
			perceptron.trainPerceptron();
			System.out.println("Training successful!");
			
			accuracy = perceptron.calculateAccuracy(testingHamDir,testingSpamDir);
			System.out.println("Overall accuracy of Perceptron without using stopwords = " + accuracy + "%\n\n");
			
			System.out.println("===Training Perceptron model started using stopWords===");
			PerceptronModel perceptronWithoutSW = new PerceptronModel(trainingHamDir, trainingSpamDir,stopWord);
			perceptronWithoutSW.trainPerceptron();
			System.out.println("Trained successful!");
			
			accuracySW = perceptronWithoutSW.calculateAccuracy(testingHamDir,testingSpamDir);
			System.out.println("Overall Accuracy of Perceptron using stopwords = " + accuracySW + "%");
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
