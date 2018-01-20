package org.dzianish.nmist;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

public class NNConfig {
//    @Getter @Setter @Wither
    private String name;
//    @Getter @Setter @Wither
    private MultiLayerConfiguration configuration;

    public NNConfig() {}


    public NNConfig withName(String name){
        setName(name);
        return this;
    }

    public NNConfig withConfiguration(MultiLayerConfiguration configuration){
        setConfiguration(configuration);
        return this;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public MultiLayerConfiguration getConfiguration() {
        return configuration;
    }

    public void setConfiguration(MultiLayerConfiguration configuration) {
        this.configuration = configuration;
    }
}
