package org.dzianish.demo.mnist.repositories.impl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.dzianish.demo.mnist.domain.NNModel;
import org.dzianish.demo.mnist.services.impl.NNConfigFactory;
import org.dzianish.demo.mnist.repositories.IModelRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Repository;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

@Repository
public class NNModelRepository implements IModelRepository {
	private static final Logger LOG = LoggerFactory.getLogger(NNConfigFactory.class);

	private static final String MODELS_REPO_PATH = "models/";
	private static final String MODEL_EXT = "/bestModel.bin";

	public NNModelRepository() {
		boolean isOK = new File(MODELS_REPO_PATH).mkdirs();
		if (isOK) {
			LOG.warn("Can't create folder to persist NN models, it may cause invalid behaviour");
		}
	}

	@Override
	public void persist(NNModel model) {
		String path = createPath(model.getName());
		try {
			ModelSerializer.writeModel(model.getModel(), path, true);
		} catch (IOException e) {
			LOG.error("Can't persist model to " + path, e);
			throw new RuntimeException(e); // TODO:
		}
	}

	@Override
	public NNModel load(String name) {
		String path = createPath(name);
		MultiLayerNetwork model;
		try {
			model = ModelSerializer.restoreMultiLayerNetwork(path);
		} catch (IOException e) {
			LOG.error("Can't load model from " + path, e);
			throw new RuntimeException(e);// TODO:
		}

		return new NNModel()
				.withName(name)
				.withModel(model);
	}

	@Override
	public boolean isModelExists(String name) {
		File model = new File(createPath(name));
		return model.exists();
	}

	@Override
	public List<String> getAvailableModelNames() {
		File folder = new File(MODELS_REPO_PATH);
		return Stream.of(folder.list())
				.filter(str -> new File(folder.getAbsolutePath(), str + MODEL_EXT).exists())
				.collect(toList());
	}

	private String createPath(String name) {
		return MODELS_REPO_PATH + name + MODEL_EXT;
	}

}
