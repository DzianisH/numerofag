package org.demo.mnist.repositories;

import org.demo.mnist.domain.NNModel;

import java.util.List;

public interface IModelRepository {
	void persist(NNModel model);

	NNModel load(String name);

	boolean isModelExists(String name);

	List<String> getAvailableModelNames();
}
