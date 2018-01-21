package org.dzianish.nmist;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class NNExecutorService {
    private static final Logger LOG = LoggerFactory.getLogger(NNExecutorService.class);

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

    public int getPredictionClass(NNModel nnModel, INDArray features) {
        MultiLayerNetwork model = nnModel.getModel();
        INDArray output = model.output(features);

        return getPredictionClass(output);
    }

    private int getPredictionClass(INDArray output) {
        INDArray prediction = output.argMax(1);
        if (LOG.isInfoEnabled()) {
            logOutput(output);
        }

        return prediction.getInt(0);
    }

    private void logOutput(INDArray output) {
        StringBuffer sb = new StringBuffer(50);
        for (int i = 0; i < output.length(); ++i) {
            double val = output.getDouble(i);
            sb.append(String.format("%.2f ", val));
        }
        LOG.info(sb.toString());
    }
}
