package org.demo.mnist.services;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.demo.mnist.consts.Constants;
import org.demo.mnist.dl4j.LocalFileModelFullInfoSaver;
import org.demo.mnist.dl4j.GenericCalculator;
import org.demo.mnist.domain.NNConfig;
import org.demo.mnist.domain.NNModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//@Service
public class NNTrainerService {
    private static final Logger LOG = LoggerFactory.getLogger(NNTrainerService.class);

    public NNModel fitModel(NNConfig config, DataSetIterator trainDS, DataSetIterator testDS) {
        MultiLayerNetwork configuration = new MultiLayerNetwork(config.getConfiguration());
        NNModel model = new NNModel()
                .withName(config.getName())
                .withModel(configuration);

        return fitModel(model, trainDS, testDS);
    }


    public NNModel fitModel(NNModel model, DataSetIterator trainDS, DataSetIterator testDS) {
        LOG.info("Initializing model " + model.getName());
        model.getModel().init();

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(
//                        new MaxEpochsTerminationCondition(MAX_EPOCHS),
                        new ScoreImprovementEpochTerminationCondition(Constants.MAX_EPOCHS_WO_IMPROVEMENT))
                .iterationTerminationConditions(new InvalidScoreIterationTerminationCondition())
                .saveLastModel(true)
                .scoreCalculator(new GenericCalculator(testDS, Evaluation::accuracy))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelFullInfoSaver("models/" + model.getName()))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,
                model.getModel(), trainDS);

        LOG.info("Training model: " + model.getName() +  " with " + model.getModel().numParams() + " params");
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        LOG.info("Termination reason: " + result.getTerminationReason());
        LOG.info("Total epochs: " + result.getTotalEpochs());
        LOG.info("Best epoch number: " + result.getBestModelEpoch());
        LOG.info("Score at best epoch: " + result.getBestModelScore());

		model.setModel(result.getBestModel());
        evaluateModel(model, testDS);

        return model;
    }

    private void evaluateModel(NNModel nnModel, DataSetIterator ds) {
        LOG.info("Evaluating model " + nnModel.getName());
        MultiLayerNetwork model = nnModel.getModel();
        ds.reset();
        Evaluation eval = new Evaluation(Constants.CLASSES); //create an evaluation object with 10 possible classes
        while (ds.hasNext()) {
            DataSet next = ds.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        LOG.info(eval.stats());
    }
}
