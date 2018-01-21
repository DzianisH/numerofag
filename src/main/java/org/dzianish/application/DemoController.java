package org.dzianish.application;

import org.dzianish.nmist.NNExecutorService;
import org.dzianish.nmist.NNModel;
import org.dzianish.nmist.NNModelRepository;
import org.dzianish.nmist.NNPredictions;
import org.dzianish.utils.Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;

import java.io.IOException;

import static org.dzianish.utils.Utils.toINDArray;
import static org.slf4j.LoggerFactory.getLogger;

@Controller
public class DemoController {
    private static final Logger LOG = getLogger(DemoController.class);

    @Autowired
    private NNExecutorService executor;
    @Autowired
    private NNModelRepository repository;


    @GetMapping("/")
    public ModelAndView getIndexPage(ModelMap model) {
        model.put("models", repository.getAvailableModelNames());
        return new ModelAndView("index", model);
    }

    @GetMapping("/demo/{modelName}")
    public ModelAndView getDemoPage(@PathVariable String modelName, ModelMap model) throws IOException {
        if (repository.isModelExists(modelName)) {
            model.put("modelName", modelName);
            return new ModelAndView("demo", model);
        }

        return new ModelAndView("redirect:/index");
    }

    @PostMapping(value = "/demo/{modelName}")
    @ResponseBody
    public NNPredictions getPrediction(@RequestBody double[][] features, @PathVariable String modelName) {
        if (LOG.isDebugEnabled()) {
            LOG.debug(Utils.toString(features));
        }

        NNModel model = repository.load(modelName);
        INDArray indFeatures = toINDArray(features);

        return executor.getPrediction(model, indFeatures);
    }
}
