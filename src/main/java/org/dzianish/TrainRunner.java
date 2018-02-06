package org.dzianish;

import static org.dzianish.consts.Constants.BATCH_SIZE;
import static org.dzianish.consts.Constants.RND_SEED;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.dzianish.domain.NNConfig;
import org.dzianish.domain.NNModel;
import org.dzianish.services.NNConfigFactory;
import org.dzianish.services.NNTrainerService;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TrainRunner {
    private static final Logger LOG = LoggerFactory.getLogger(TrainRunner.class);

    public static void main(String[] args) throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(BATCH_SIZE, true, RND_SEED);
        DataSetIterator mnistTest = new MnistDataSetIterator(BATCH_SIZE, false, RND_SEED);

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
