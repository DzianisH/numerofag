package org.dzianish;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.dzianish.nmist.*;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class Runner {
    private static final Logger LOG = LoggerFactory.getLogger(Runner.class);
    
    public static void main(String[] args) throws IOException {
        int batchSize = 256, seed = 132;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        LOG.info("Building model..");
        NNConfig conf = new NNConfigFactory().createSingleLayerModel();

        LOG.info("Initializing model..");
        NNModel model = new NNTrainer().fitModel(conf, mnistTrain, mnistTest);

        LOG.info("Saving model..");
        new NNModelRepository().persist(model);


    }
}
