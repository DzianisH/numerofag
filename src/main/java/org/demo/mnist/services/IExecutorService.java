package org.demo.mnist.services;

import org.demo.mnist.domain.NNModel;
import org.demo.mnist.domain.NNPredictions;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface IExecutorService {
	NNPredictions getPrediction(NNModel nnModel, INDArray features);

	int getPredictionClass(NNModel nnModel, INDArray features);
}
