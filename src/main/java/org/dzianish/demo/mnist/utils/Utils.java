package org.dzianish.demo.mnist.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

abstract public class Utils {
    public static INDArray toINDArray(double[][] array) {
        return new NDArray(array).reshape(1, array.length * array[0].length);
    }

    public static INDArray toINDArray(double[] array){
        double newArray[][] = new double[1][array.length];
        System.arraycopy(array, 0, newArray[0], 0, array.length);
        return new NDArray(newArray);
    }

    public static String toString(double[][] features) {
        StringBuilder sb = new StringBuilder(features.length * features.length * 5 + features.length + 11);

        sb.append("FEATURES:\n");
        for (double[] featuresLine : features) {
            for (double feature : featuresLine) {
                sb.append(String.format("%.2f ", feature));
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}