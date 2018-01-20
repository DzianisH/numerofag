package org.dzianish;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.dzianish.nmist.NNModel;
import org.dzianish.nmist.NNModelRepository;
import org.dzianish.nmist.NNTrainer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class LoadRunner {
    private static final Logger LOG = LoggerFactory.getLogger(LoadRunner.class);

    public static void main(String[] args) throws IOException {
        int batchSize = 256, seed = 132;
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        LOG.info("Loading model..");
        NNModel model = new NNModelRepository().load("single-layer-model");


        LOG.info("Evaluating model..");
        new NNTrainer().evaluateModel(model, mnistTest);
    }
}
