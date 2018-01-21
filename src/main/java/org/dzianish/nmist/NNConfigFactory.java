package org.dzianish.nmist;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.deeplearning4j.nn.api.OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
import static org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.MAX;
import static org.deeplearning4j.nn.weights.WeightInit.XAVIER;
import static org.dzianish.nmist.Constants.*;
import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MCXENT;

public class NNConfigFactory {
    // 39760 params 50890
    public NNConfig createSingleLayerConfig() {
        return new NNConfig()
                .withName(SINGLE_LAYER_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.9))
                        .learningRate(0.01)
                        .regularization(true).l2(0.32)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(INPUT_COLS * INPUT_ROWS)
                                .nOut(64)
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

    // 84060 params
    public NNConfig createTwoLayerConfig() {
        return new NNConfig()
                .withName(TWO_LAYER_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.9))
                        .learningRate(0.01)
                        .regularization(true).l2(0.5)
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

    // TODO: Doesn't work
    // 93256 params?
    public NNConfig createConvolutionConfig() {
        return new NNConfig()
                .withName(CONVOLUTION_MODEL)
                .withConfiguration(new NeuralNetConfiguration.Builder()
                        .iterations(ITERATIONS)
                        .seed(RND_SEED)
                        .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.86))
                        .learningRate(0.05)
                        .regularization(true).l2(0.9)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(2, 2)
                                .nIn(INPUT_COLS * INPUT_ROWS)
                                .nOut(8)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(3, 3)
                                .stride(1, 1)
                                .poolingType(MAX)
                                .build())
                        .layer(2, new DenseLayer.Builder()
                                .nIn(100500)
                                .nOut(50)
                                .activation(RELU)
                                .weightInit(XAVIER)
                                .build())
                        .layer(3, new OutputLayer.Builder()
                                .nIn(100500)
                                .nOut(CLASSES)
                                .activation(SOFTMAX)
                                .weightInit(XAVIER)
                                .build())
                        .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(INPUT_COLS, INPUT_ROWS, INPUT_DEPTH))
//                        .setInputType(InputType.feedForward(INPUT_COLS * INPUT_ROWS))
                        .setInputType(InputType.convolutionalFlat(28, 28, 1))
                        .pretrain(false)
                        .backprop(true)
                        .build());
    }
}
