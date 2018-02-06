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
import static org.deeplearning4j.nn.weights.WeightInit.XAVIER_UNIFORM;
import static org.dzianish.consts.Constants.CLASSES;
import static org.dzianish.consts.Constants.INPUT_COLS;
import static org.dzianish.consts.Constants.INPUT_DEPTH;
import static org.dzianish.consts.Constants.INPUT_ROWS;
import static org.dzianish.consts.Constants.ITERATIONS;
import static org.dzianish.consts.Constants.RND_SEED;
import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MCXENT;

public class NNConfigFactory {

	public NNConfig createTwoLayerConfig() {
		return new NNConfig()
				.withName("D100-O")
				.withConfiguration(new NeuralNetConfiguration.Builder()
						.iterations(ITERATIONS)
						.seed(RND_SEED)
						.optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
						.updater(new Nesterovs(0.9))
						.learningRate(0.145)
						.regularization(true).l2(6e-3)
//                        .dropOut(0.1)
						.list()
						.layer(0, new DenseLayer.Builder()
								.nIn(INPUT_COLS * INPUT_ROWS)
								.nOut(100)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
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
				.withName("D100-D32-O")
				.withConfiguration(new NeuralNetConfiguration.Builder()
						.iterations(ITERATIONS)
						.seed(RND_SEED)
						.optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
						.updater(new Nesterovs(0.9))
						.learningRate(0.145)
						.regularization(true).l2(9e-2)
						.list()
						.layer(0, new DenseLayer.Builder()
								.nIn(INPUT_COLS * INPUT_ROWS)
								.nOut(100)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.build())
						.layer(1, new DenseLayer.Builder()
								.nOut(32)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
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

	public NNConfig createConvolutionConfig() {
		return new NNConfig()
				.withName("C12-C12S-D90-O")
				.withConfiguration(new NeuralNetConfiguration.Builder()
						.iterations(ITERATIONS)
						.seed(RND_SEED)
						.optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
						.updater(new Nesterovs(0.9))
						.learningRate(0.13)//
						.regularization(true).l2(1e-1)
//						.dropOut(1e-6)
						.list()
						.layer(0, new ConvolutionLayer.Builder()
								.nIn(1)
								.nOut(12)
								.kernelSize(3, 3)
								.stride(2, 2)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.build())
						.layer(1, new ConvolutionLayer.Builder()
								.nOut(12)
								.kernelSize(2, 2)
								.stride(1, 1)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.build())
						.layer(2, new SubsamplingLayer.Builder()
								.kernelSize(2, 2)
								.poolingType(MAX)
								.build())
						.layer(3, new DenseLayer.Builder()
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.nOut(90)
								.build())
						.layer(4, new OutputLayer.Builder()
								.nOut(CLASSES)
								.activation(SOFTMAX)
								.weightInit(XAVIER)
								.build())
						.inputPreProcessor(0, new FeedForwardToCnnPreProcessor(INPUT_COLS, INPUT_ROWS, INPUT_DEPTH))
						.setInputType(convolutionalFlat(INPUT_COLS, INPUT_ROWS, INPUT_DEPTH))
						.pretrain(false)
						.backprop(true)
						.build());
	}

	public NNConfig createDeepConvolutionConfig() {
		return new NNConfig()
				.withName("C12-C12S-D100-D32-O")
				.withConfiguration(new NeuralNetConfiguration.Builder()
						.iterations(ITERATIONS)
						.seed(RND_SEED)
						.optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
						.updater(new Nesterovs(0.9))
						.learningRate(0.13)//
						.regularization(true).l2(0.6)
//						.dropOut(1e-6)
						.list()
						.layer(0, new ConvolutionLayer.Builder()
								.nIn(1)
								.nOut(12)
								.kernelSize(3, 3)
								.stride(2, 2)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.build())
						.layer(1, new ConvolutionLayer.Builder()
								.nOut(8)
								.kernelSize(2, 2)
								.stride(1, 1)
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.build())
						.layer(2, new SubsamplingLayer.Builder()
								.kernelSize(2, 2)
								.poolingType(MAX)
								.build())
						.layer(3, new DenseLayer.Builder()
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.nOut(90)
								.build())
						.layer(4, new DenseLayer.Builder()
								.activation(RELU)
								.weightInit(XAVIER_UNIFORM)
								.nOut(32)
								.build())
						.layer(5, new OutputLayer.Builder()
								.nOut(CLASSES)
								.activation(SOFTMAX)
								.weightInit(XAVIER)
								.build())
						.inputPreProcessor(0, new FeedForwardToCnnPreProcessor(INPUT_COLS, INPUT_ROWS, INPUT_DEPTH))
						.setInputType(convolutionalFlat(INPUT_COLS, INPUT_ROWS, INPUT_DEPTH))
						.pretrain(false)
						.backprop(true)
						.build());
	}
}
