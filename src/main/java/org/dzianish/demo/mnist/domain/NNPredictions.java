package org.dzianish.demo.mnist.domain;

public class NNPredictions {
    private double[] estimates;
    private int predictionClass;

    public NNPredictions withPredictionsCLass(int n){
        setPredictionClass(n);
        return this;
    }

    public NNPredictions withEstimates(double estimates[]){
        setEstimates(estimates);
        return this;
    }

    public int getPredictionClass() {
        return predictionClass;
    }

    public void setPredictionClass(int predictionClass) {
        this.predictionClass = predictionClass;
    }

    public double[] getEstimates() {
        return estimates;
    }

    public void setEstimates(double[] estimates) {
        this.estimates = estimates;
    }
}
