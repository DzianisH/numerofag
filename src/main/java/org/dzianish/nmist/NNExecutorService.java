package org.dzianish.nmist;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;

@Service
public class NNExecutorService {

    public int getPrediction(NNModel nnModel, INDArray features){
        MultiLayerNetwork model = nnModel.getModel();
        INDArray output = model.output(features);
        INDArray prediction = output.argMax(1);

        return prediction.getInt(0);
    }
}
