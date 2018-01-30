package org.dzianish.services;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.dzianish.GenericCalculator;
import org.dzianish.domain.NNConfig;
import org.dzianish.domain.NNModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.dzianish.consts.Constants.*;

//@Service
public class NNTrainerService {
    private static final Logger LOG = LoggerFactory.getLogger(NNTrainerService.class);

    public NNModel fitModel(NNConfig config, DataSetIterator trainDS, DataSetIterator testDS) {

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(
                        new MaxEpochsTerminationCondition(MAX_EPOCHS),
                        new ScoreImprovementEpochTerminationCondition(MAX_EPOCHS_WO_IMPROVEMENT))
                .iterationTerminationConditions(new InvalidScoreIterationTerminationCondition())
                .saveLastModel(true)
                .scoreCalculator(new GenericCalculator(testDS, Evaluation::accuracy))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("models/" + config.getName()))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,
                new MultiLayerNetwork(config.getConfiguration()), trainDS);

        LOG.info("Training model: " + config.getName());

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        LOG.info("Termination reason: " + result.getTerminationReason());
        LOG.info("Total epochs: " + result.getTotalEpochs());
        LOG.info("Best epoch number: " + result.getBestModelEpoch());
        LOG.info("Score at best epoch: " + result.getBestModelScore());

        NNModel nnModel = new NNModel()
                .withName(config.getName())
                .withModel(result.getBestModel());

        evaluateModel(nnModel, testDS);

        return nnModel;
    }

    private void evaluateModel(NNModel nnModel, DataSetIterator ds) {
        LOG.info("Evaluating model " + nnModel.getName());
        MultiLayerNetwork model = nnModel.getModel();
        ds.reset();
        Evaluation eval = new Evaluation(CLASSES); //create an evaluation object with 10 possible classes
        while (ds.hasNext()) {
            DataSet next = ds.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        LOG.info(eval.stats());
    }
}
