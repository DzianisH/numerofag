package org.dzianish.services;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.dzianish.domain.NNConfig;
import org.nd4j.linalg.learning.config.Nesterovs;

import static org.deeplearning4j.nn.api.OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
import static org.deeplearning4j.nn.conf.inputs.InputType.convolutionalFlat;
import static org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.MAX;
import static org.deeplearning4j.nn.weights.WeightInit.XAVIER;
import static org.dzianish.consts.Constants.*;
import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MCXENT;

public class NNConfigFactory {

    public NNConfig createTwoLayerConfig() {
        return new NNConfig()
                .withName(TWO_LAYER_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.9))
                        .learningRate(0.145)
                        .regularization(true).l2(0.006)
//                        .dropOut(0.1)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(INPUT_COLS * INPUT_ROWS)
                                .nOut(16)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(1, new OutputLayer.Builder()
                                .nOut(CLASSES)
                                .activation(SOFTMAX)
                                .weightInit(XAVIER)
                                .lossFunction(MCXENT)
                                .build())
                        .pretrain(false)
                        .backprop(true)
                        .build());
    }

    public NNConfig createThreeLayerConfig() {
        return new NNConfig()
                .withName(TREE_LAYER_MODEL + 1)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.9))
                        .learningRate(0.145)
                        .regularization(true).l2(0.6)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(INPUT_COLS * INPUT_ROWS)
                                .nOut(78)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(1, new DenseLayer.Builder()
                                .nOut(16)
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

    // TODO: Doesn't work
    public NNConfig createConvolutionConfig() {
        return new NNConfig()
                .withName(CONVOLUTION_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.9))
                        .learningRate(0.145)
                        .regularization(true).l2(0.06)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(3, 3)
                                .nIn(1)
                                .nOut(8)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(2, 2)
                                .stride(1, 1)
                                .poolingType(MAX)
                                .build())
                        .layer(2, new OutputLayer.Builder()
                                .nOut(CLASSES)
                                .activation(SOFTMAX)
                                .weightInit(XAVIER)
                                .build())
                        .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(INPUT_COLS, INPUT_ROWS, INPUT_DEPTH))
                        .setInputType(convolutionalFlat(28, 28, 1))
                        .pretrain(false)
                        .backprop(true)
                        .build());
    }
}
