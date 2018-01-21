package org.dzianish.application;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.servlet.ModelAndView;

import java.util.HashMap;

@Controller("/")
public class RootController {

    @GetMapping
    public ModelAndView getIndexPage(){
        return new ModelAndView("index", new HashMap<>());
    }
}
