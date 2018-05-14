package org.demo.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.demo.mnist.consts.Constants;
import org.demo.mnist.domain.NNConfig;
import org.demo.mnist.domain.NNModel;
import org.demo.mnist.services.NNConfigFactory;
import org.demo.mnist.services.NNTrainerService;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TrainRunner {
    private static final Logger LOG = LoggerFactory.getLogger(TrainRunner.class);

    public static void main(String[] args) throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(Constants.BATCH_SIZE, true, Constants.RND_SEED);
        DataSetIterator mnistTest = new MnistDataSetIterator(Constants.BATCH_SIZE, false, Constants.RND_SEED);

        LOG.info("Building model..");
        NNConfig conf = new NNConfigFactory().createDeepConvolutionConfig();
        NNModel model = new NNTrainerService().fitModel(conf, mnistTrain, mnistTest);

//        LOG.info("Loading..");
//        NNModel model = new NNModelRepository().load(TREE_LAYER_MODEL);
//        model = new NNTrainerService().fitModel(model, mnistTrain, mnistTest);

//        LOG.info("Saving model..");
//        new NNModelRepository().persist(model);


    }
}
