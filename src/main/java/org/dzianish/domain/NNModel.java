package org.dzianish.domain;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class NNModel {
    private String name;
    private MultiLayerNetwork model;

    public NNModel withName(String name) {
        setName(name);
        return this;
    }

    public NNModel withModel(MultiLayerNetwork configuration) {
        setModel(configuration);
        return this;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }
}
