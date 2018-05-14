package org.dzianish.demo.mnist.dl4j;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static java.lang.String.format;

/**
 * Use EarlyStoppingListener
 */
@Deprecated
public class LocalFileModelFullInfoSaver extends LocalFileModelSaver {
    private static final Logger LOG = LoggerFactory.getLogger(LocalFileModelFullInfoSaver.class);

    private final String directory;

    public LocalFileModelFullInfoSaver(String directory) {
        super(directory);
        this.directory = directory;
    }

    @Override
    public void saveBestModel(MultiLayerNetwork net, double score) throws IOException {
        super.saveBestModel(net, score);
        saveInfo("bestModel.info", score);
    }

    @Override
    public void saveLatestModel(MultiLayerNetwork net, double score) throws IOException {
        super.saveLatestModel(net, score);
        saveInfo("latestModel.info", score);
    }

    private void saveInfo(String fileName, double score) {
        String fileAbsoluteName = FilenameUtils.concat(directory, fileName);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileAbsoluteName))) {
            writer.write(
                    format("score = %.18f\nerror = %.2f%%\naccuracy = %.2f%%\nepoch = unknown\n",
                            score, score * 100, 100 * (1 - score)));
            writer.flush();
        } catch (IOException e) {
            LOG.warn("Can't save model secondary info", e);
        }
    }
}
