package org.dzianish.application;

import org.dzianish.nmist.NNExecutorService;
import org.dzianish.nmist.NNInput;
import org.dzianish.nmist.NNModel;
import org.dzianish.nmist.NNModelRepository;
import org.dzianish.utils.Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import java.util.HashMap;

@Controller
public class DemoController {
    private static final Logger LOG = LoggerFactory.getLogger(DemoController.class);

    @Autowired
    private NNExecutorService executor;
    @Autowired
    private NNModelRepository repository;

    @GetMapping("/demo")
    public ModelAndView getIndexPage(){
        return new ModelAndView("index", new HashMap<>());
    }

    @PostMapping("/demo")
    @ResponseBody
    public Integer getPrediction(@RequestBody NNInput input) {
        if(LOG.isDebugEnabled()) {
            LOG.debug(Utils.toString(input.getFeatures()));
        }

        NNModel model = repository.load(input.getModel());
        INDArray features = Utils.toINDArray(input.getFeatures());

        return executor.getPrediction(model, features);
    }
}
