package org.dzianish.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

abstract public class Utils {
    public static INDArray toINDArray(double[][] array) {
        return new NDArray(array).reshape(1, array.length * array[0].length);
    }
}
