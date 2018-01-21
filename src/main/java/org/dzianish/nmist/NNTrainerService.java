package org.dzianish.nmist;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.dzianish.nmist.Constants.CLASSES;
import static org.dzianish.nmist.Constants.EPOCHS;

//@Service
public class NNTrainerService {
    private static final Logger LOG = LoggerFactory.getLogger(NNTrainerService.class);

    public NNModel fitModel(NNConfig config, DataSetIterator trainDS, DataSetIterator testDS) {
        MultiLayerConfiguration configuration = config.getConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        LOG.info("Number of parameters: " + model.numParams());

        //print the score with every 10 iteration
        model.setListeners(new ScoreIterationListener(10));

        LOG.info("Training model " + config.getName());
        for (int i = 0; i < EPOCHS; ++i) {
            LOG.info("Starting epoch №" + i);
            model.fit(trainDS);
        }

        NNModel nnModel = new NNModel()
                .withName(config.getName())
                .withModel(model);

        evaluateModel(nnModel, testDS);

        return nnModel;
    }

    private double evaluateModel(NNModel nnModel, DataSetIterator ds) {
        LOG.info("Evaluating model " + nnModel.getName());
        MultiLayerNetwork model = nnModel.getModel();
        Evaluation eval = new Evaluation(CLASSES); //create an evaluation object with 10 possible classes
        while (ds.hasNext()) {
            DataSet next = ds.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        LOG.info(eval.stats());
        return eval.f1();
    }
}
