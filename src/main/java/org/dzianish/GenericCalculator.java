package org.dzianish;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.function.ToDoubleFunction;

import static org.dzianish.consts.Constants.CLASSES;

public class GenericCalculator implements ScoreCalculator<MultiLayerNetwork> {
    private final DataSetIterator dsIterator;
    private ToDoubleFunction<Evaluation> extractor;

    public GenericCalculator(DataSetIterator dsIterator, ToDoubleFunction<Evaluation> extractor){
        this.dsIterator = dsIterator;
        this.extractor = extractor;
    }

    @Override
    public double calculateScore(MultiLayerNetwork network) {
        dsIterator.reset();

        Evaluation eval = new Evaluation(CLASSES); //create an evaluation object with 10 possible classes
        while (dsIterator.hasNext()) {
            DataSet next = dsIterator.next();
            INDArray output = network.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        return 1 - extractor.applyAsDouble(eval);
    }
}
