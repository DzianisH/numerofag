package org.dzianish.demo.mnist.dl4j;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.dzianish.demo.mnist.consts.Constants;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.ToDoubleFunction;

public class GenericCalculator implements ScoreCalculator<MultiLayerNetwork> {
	private static final Logger LOG = LoggerFactory.getLogger(GenericCalculator.class);

	private final DataSetIterator dsIterator;
	private ToDoubleFunction<Evaluation> extractor;

	public GenericCalculator(DataSetIterator dsIterator, ToDoubleFunction<Evaluation> extractor) {
		this.dsIterator = dsIterator;
		this.extractor = extractor;
	}

	@Override
	public double calculateScore(MultiLayerNetwork network) {
		dsIterator.reset();

		Evaluation eval = new Evaluation(Constants.CLASSES); //create an evaluation object with 10 possible classes
		while (dsIterator.hasNext()) {
			DataSet next = dsIterator.next();
			INDArray output = network.output(next.getFeatureMatrix()); //get the networks prediction
			eval.eval(next.getLabels(), output); //check the prediction against the true class
		}

		double score = 1 - extractor.applyAsDouble(eval);
		LOG.info("Score: " + score);
		return score;
	}
}
