package org.dzianish;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.dzianish.nmist.NNModel;
import org.dzianish.nmist.NNModelRepository;
import org.dzianish.nmist.NNTrainer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static org.dzianish.nmist.Constants.BATCH_SIZE;
import static org.dzianish.nmist.Constants.RND_SEED;

public class LoadRunner {
    private static final Logger LOG = LoggerFactory.getLogger(LoadRunner.class);

    public static void main(String[] args) throws IOException {
        DataSetIterator mnistTest = new MnistDataSetIterator(BATCH_SIZE, false, RND_SEED);

        LOG.info("Loading model..");
        NNModel model = new NNModelRepository().load("single-layer-model");


        LOG.info("Evaluating model..");
        new NNTrainer().evaluateModel(model, mnistTest);
    }
}
