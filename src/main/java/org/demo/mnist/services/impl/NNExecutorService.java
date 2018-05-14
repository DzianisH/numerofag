package org.demo.mnist.services.impl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.demo.mnist.domain.NNModel;
import org.demo.mnist.domain.NNPredictions;
import org.demo.mnist.services.IExecutorService;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class NNExecutorService implements IExecutorService {
    private static final Logger LOG = LoggerFactory.getLogger(NNExecutorService.class);

    @Override
    public NNPredictions getPrediction(NNModel nnModel, INDArray features) {
        MultiLayerNetwork model = nnModel.getModel();
        INDArray output = model.output(features);

        double[] estimates = new double[output.size(1)];
        for (int i = 0; i < estimates.length; ++i) {
            estimates[i] = output.getDouble(i);
        }

        return new NNPredictions()
                .withEstimates(estimates)
                .withPredictionsCLass(getPredictionClass(output));
    }

    @Override
    public int getPredictionClass(NNModel nnModel, INDArray features) {
        MultiLayerNetwork model = nnModel.getModel();
        INDArray output = model.output(features);

        return getPredictionClass(output);
    }

    private int getPredictionClass(INDArray output) {
        INDArray prediction = output.argMax(1);
        if (LOG.isDebugEnabled()) {
            logOutput(output);
        }

        return prediction.getInt(0);
    }

    private void logOutput(INDArray output) {
        StringBuilder sb = new StringBuilder(50);
        for (int i = 0; i < output.length(); ++i) {
            double val = output.getDouble(i);
            sb.append(String.format("%.2f ", val));
        }
        LOG.debug(sb.toString());
    }
}
