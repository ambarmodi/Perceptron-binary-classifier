import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map.Entry;

/**
 * @author Ambar
 *
 */
public class PerceptronModel {
	private static final String SPAM_CLASS = "spam";
	private static final String HAM_CLASS = "ham";
	private static final int SPAM = -1;
	private static final int HAM = -2;
	private static final double HAM_OUTPUT = -1;
	private static final double SPAM_OUTPUT = 1;
	private static final String CLASS = "CLASS";
	private static final String SEPERATOR = " ";
	
	/* bias weight w0*/
	private double biasWeight = 0.0; 

	public static int iterations; /* Iterations for convergence */
	public static double eta; /* This is the learning rate for perceptron */

	/* Weights will be of size of vocab. */
	HashMap<String, Double> weights = new HashMap<String, Double>();

	/* If we are using the stop words list for optimization */
	boolean usingStopWords = false;

	LinkedHashSet<String> stopWords = new LinkedHashSet<String>();
	LinkedHashSet<String> vocab = new LinkedHashSet<String>();

	/* Holds the entire example and feature matrix. */
	ArrayList<HashMap<String, Integer>> data = new ArrayList<HashMap<String, Integer>>();

	/**
	 * Constructor when not using stopwords
	 * 
	 * @throws IOException
	 */
	public PerceptronModel(String trainingHamDir, String trainingSpamDir)
			throws IOException {
		createVocab(trainingHamDir, trainingSpamDir);
		readHamDir(trainingHamDir);
		readSpamDir(trainingSpamDir);
	}

	/**
	 * Constructor when not using stopwords
	 * 
	 * @throws IOException
	 */
	public PerceptronModel(String trainingHamDir, String trainingSpamDir,
			String stopWords) throws IOException {
		this.usingStopWords = true;
		readStopWords(stopWords);
		createVocab(trainingHamDir, trainingSpamDir);
		readHamDir(trainingHamDir);
		readSpamDir(trainingSpamDir);
	}

	/**
	 * Reads the stop words
	 * 
	 * @param stopWords
	 * @throws IOException
	 */
	private void readStopWords(String stopWords) throws IOException {
		File file = new File(stopWords);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line = null;
		while ((line = br.readLine()) != null) {
			this.stopWords.add(line);
		}
		br.close();
	}

	/**
	 * Reads the SPAM Directory.
	 * 
	 * @throws IOException
	 */
	private void readSpamDir(String trainingSpamDir) throws IOException {
		File spamDir = new File(trainingSpamDir);
		String line = null;
		for (File sFile : spamDir.listFiles()) {
			HashMap<String, Integer> example = new HashMap<String, Integer>();
			FileReader fr = new FileReader(sFile);
			BufferedReader br = new BufferedReader(fr);
			while ((line = br.readLine()) != null) {
				String list[] = line.split(SEPERATOR);
				for (String word : list) {
					if (usingStopWords && (stopWords.contains(word))) {
						// Do Nothing
					} else {
						if (!example.containsKey(word)) {
							example.put(word, 1);
						} else {
							example.put(word, example.get(word) + 1);
						}
					}
				}
			}
			br.close();
			example.put(CLASS, SPAM);
			data.add(example);
		}
	}

	/**
	 * Reads the HAM Directory.
	 * 
	 * @throws IOException
	 */

	private void readHamDir(String trainingHamDir) throws IOException {
		File hamDir = new File(trainingHamDir);
		String line = null;

		for (File hFile : hamDir.listFiles()) {
			HashMap<String, Integer> example = new HashMap<String, Integer>();
			FileReader fr = new FileReader(hFile);
			BufferedReader br = new BufferedReader(fr);
			while ((line = br.readLine()) != null) {
				String list[] = line.split(SEPERATOR);
				for (String word : list) {
					if (usingStopWords && stopWords.contains(word)) {
						// Do Nothing
					} else {
						if (!example.containsKey(word)) {
							example.put(word, 1);
						} else {
							example.put(word, example.get(word) + 1);
						}
					}
				}
			}
			br.close();
			example.put(CLASS, HAM);
			data.add(example);
		}
	}

	/**
	 * Creates the vocabulary for entire model
	 * 
	 * @param trainingHamDir
	 * @param trainingSpamDir
	 * @throws IOException
	 */
	private void createVocab(String trainingHamDir, String trainingSpamDir)
			throws IOException {
		File hamDir = new File(trainingHamDir);
		String line = null;
		for (File hFile : hamDir.listFiles()) {
			FileReader fr = new FileReader(hFile);
			BufferedReader br = new BufferedReader(fr);
			while ((line = br.readLine()) != null) {
				String list[] = line.split(SEPERATOR);
				for (String word : list) {
					if (usingStopWords && stopWords.contains(word)) {
						// Do Nothing
					} else {
						this.vocab.add(word);

					}
				}
			}
			br.close();
		}

		File spamDir = new File(trainingSpamDir);
		for (File sFile : spamDir.listFiles()) {
			FileReader fr = new FileReader(sFile);
			BufferedReader br = new BufferedReader(fr);
			while ((line = br.readLine()) != null) {
				String list[] = line.split(SEPERATOR);
				for (String word : list) {
					if (usingStopWords && (stopWords.contains(word))) {
						// Do Nothing
					} else {
						vocab.add(word);
					}
				}
			}
			br.close();
		}

	}

	/**
	 * This Method trains the Perceptron Model
	 */
	public void trainPerceptron() {
		double actualOutput = 0, predictedOutput = 0;
		for (String word : vocab) {
			weights.put(word, (2*Math.random()) - 1);			
		}
		for (int j = 0; j < iterations; j++) {
			for (int i = 0; i < data.size(); i++) {
				HashMap<String, Integer> example = data.get(i);
				actualOutput = example.get(CLASS) == HAM ? HAM_OUTPUT : SPAM_OUTPUT;
				predictedOutput = predictOutput(example);
				
				double error = (actualOutput - predictedOutput);
				
				if (error != 0) {
					for (Entry<String, Integer> entry : example.entrySet()) {
						if (!entry.getKey().equalsIgnoreCase(CLASS)) {
							double oldWeight = weights.get(entry.getKey());
							double newWeight = oldWeight
									+ (eta * error * entry.getValue());
							weights.put(entry.getKey(), newWeight);
						}
					}
					biasWeight = biasWeight + eta * error;
				}
			}
		}
	}

	/**
	 * This method predicts the output
	 * 
	 * @param example
	 * @return
	 */
	private double predictOutput(HashMap<String, Integer> example) {
		double sumOfWeights = biasWeight;
		for (Entry<String, Integer> entry : example.entrySet()) {
			String feature = entry.getKey();
			int occerence = entry.getValue();
			// System.out.println(feature);
			if (weights.get(entry.getKey()) != null)
				sumOfWeights = sumOfWeights + weights.get(feature) * occerence;
		}
		return sumOfWeights > 0 ? SPAM_OUTPUT : HAM_OUTPUT;
	}

	/**
	 * Calculate the accuracy of the Model
	 * 
	 * @throws IOException
	 */
	public double calculateAccuracy(String testingHamDir, String testingSpamDir)
			throws IOException {
		File hamDir = new File(testingHamDir);
		double hamAccuracy = 0, hamTotal = 0;
		double spamAccuracy = 0, spamTotal = 0;
		for (File doc : hamDir.listFiles()) {
			String result = applyPerceptron(doc, HAM_CLASS);
			if (result.equals(HAM_CLASS)) {
				hamAccuracy++;
			}
			hamTotal++;
		}
		System.out.println("\tAccuracy for Ham Class="
				+ (hamAccuracy / hamTotal * 100) + "%");

		File spamDir = new File(testingSpamDir);
		for (File doc : spamDir.listFiles()) {
			String result = applyPerceptron(doc, SPAM_CLASS);
			if (result.equals(SPAM_CLASS)) {
				spamAccuracy++;
			}
			spamTotal++;
		}
		System.out.println("\tAccuracy for Spam Class="
				+ (spamAccuracy / spamTotal * 100) + "%");

		return ((hamAccuracy + spamAccuracy) / (spamTotal + hamTotal) * 100);

	}

	/**
	 * For given document it predicts the output
	 * 
	 * @throws IOException
	 */
	private String applyPerceptron(File doc, String spamClass)
			throws IOException {
		String line = null;
		HashMap<String, Integer> example = new HashMap<String, Integer>();

		FileReader fr = new FileReader(doc);
		BufferedReader br = new BufferedReader(fr);
		while ((line = br.readLine()) != null) {
			String list[] = line.split(SEPERATOR);
			for (String word : list) {
				if (usingStopWords && (stopWords.contains(word))) {
					// Do Nothing
				} else {
					if (example.containsKey(word)) {
						example.put(word, example.get(word) + 1);
					} else {
						example.put(word, 1);
					}
				}
			}
		}
		br.close();
		return ((predictOutput(example) == SPAM_OUTPUT) ? SPAM_CLASS : HAM_CLASS);
	}
	

	/**
	 * For debugging purpose to print the weights
	 * 
	 * @param name
	 */
	@SuppressWarnings("unused")
	private void printWeights(String name) {
		System.out.println(name + "\n");
		try {
			PrintWriter writer;
			writer = new PrintWriter(name, "UTF-8");
			for (Entry<String, Double> entry : weights.entrySet()) {
				writer.println(entry.getValue());
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}