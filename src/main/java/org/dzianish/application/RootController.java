package org.dzianish.application;

import org.dzianish.nmist.NNExecutorService;
import org.dzianish.nmist.NNModelRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import java.util.HashMap;

@Controller
public class RootController {
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
    public Integer getPrediction() {
        return 5;
    }
}
