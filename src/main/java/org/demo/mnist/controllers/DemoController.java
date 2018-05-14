package org.demo.mnist.controllers;

import org.demo.mnist.domain.NNModel;
import org.demo.mnist.domain.NNPredictions;
import org.demo.mnist.repositories.NNModelRepository;
import org.demo.mnist.utils.Utils;
import org.demo.mnist.services.NNExecutorService;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;

import static org.demo.mnist.utils.Utils.toINDArray;
import static org.slf4j.LoggerFactory.getLogger;

@Controller
public class DemoController {
    private static final Logger LOG = getLogger(DemoController.class);

    @Autowired
    private NNExecutorService executor;
    @Autowired
    private NNModelRepository repository;


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

        return executor.getPrediction(model, indFeatures);
    }
}
