package org.dzianish.nmist;

public class NNInput {
    private String model;
    private double[][] features;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public double[][] getFeatures() {
        return features;
    }

    public void setFeatures(double[][] features) {
        this.features = features;
    }
}
