package org.dzianish.nmist;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.learning.config.Nesterovs;

import static org.deeplearning4j.nn.api.OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
import static org.deeplearning4j.nn.weights.WeightInit.XAVIER;
import static org.dzianish.nmist.Constants.*;
import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;

public class NNConfigFactory {
    public NNConfig createSingleLayerModel() {
        return new NNConfig()
                .withName(SINGLE_LAYER_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.86))
                        .learningRate(0.05)
                        .regularization(true).l2(1e-3)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(INPUT_COLS * INPUT_ROWS)
                                .nOut(50)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(1, new OutputLayer.Builder()
                                .nOut(CLASSES)
                                .activation(SOFTMAX)
                                .weightInit(XAVIER)
                                .build())
                        .pretrain(false)
                        .backprop(true)
                        .build());
    }

    public NNConfig createTwoLayerModel() {
        return new NNConfig()
                .withName(TWO_LAYER_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.86))
                        .learningRate(0.05)
                        .regularization(true).l2(1e-3)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(INPUT_COLS * INPUT_ROWS)
                                .nOut(100)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(1, new DenseLayer.Builder()
                                .nOut(50)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(2, new OutputLayer.Builder()
                                .nOut(CLASSES)
                                .activation(SOFTMAX)
                                .weightInit(XAVIER)
                                .build())
                        .pretrain(false)
                        .backprop(true)
                        .build());
    }
}
