package org.dzianish.demo.mnist.controllers;

import org.dzianish.demo.mnist.domain.NNModel;
import org.dzianish.demo.mnist.domain.NNPredictions;
import org.dzianish.demo.mnist.utils.Utils;
import org.dzianish.demo.mnist.repositories.IModelRepository;
import org.dzianish.demo.mnist.services.IExecutorService;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import static org.slf4j.LoggerFactory.getLogger;

@Controller
public class DemoController {
	private static final Logger LOG = getLogger(DemoController.class);

	@Autowired
	private IExecutorService executor;
	@Autowired
	private IModelRepository repository;


	@GetMapping({"/", "/demo"})
	public ModelAndView getIndexPage(ModelMap model) {
		model.put("models", repository.getAvailableModelNames());
		return new ModelAndView("index", model);
	}

	@GetMapping("/demo/{modelName:.+}")
	public ModelAndView getDemoPage(@PathVariable String modelName, ModelMap model) {
		if (repository.isModelExists(modelName)) {
			model.put("modelName", modelName);
			return new ModelAndView("demo", model);
		}

		return new ModelAndView("redirect:/index");
	}

	@PostMapping(value = "/demo/{modelName:.+}")
	@ResponseBody
	public NNPredictions getPrediction(@RequestBody double[][] features, @PathVariable String modelName) {
		if (LOG.isDebugEnabled()) {
			LOG.debug(Utils.toString(features));
		}

		NNModel model = repository.load(modelName);
		INDArray indFeatures = Utils.toINDArray(features);

		NNPredictions predictions = executor.getPrediction(model, indFeatures);
		LOG.info(predictions.toString());
		return predictions;
	}
}
