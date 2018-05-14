package org.demo.mnist;

import com.google.common.io.Files;
import org.demo.mnist.domain.NNModel;
import org.demo.mnist.repositories.impl.NNModelRepository;
import org.demo.mnist.services.IExecutorService;
import org.demo.mnist.services.impl.NNExecutorService;
import org.demo.mnist.utils.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.stream.Stream;

public class CSVRunner {
	private static final Logger LOG = LoggerFactory.getLogger(CSVRunner.class);

	public static void main(String[] args) throws IOException {
		LOG.info("Loading model..");
		NNModel model = new NNModelRepository().load("C16S-C32S-D128-O=L2(1e-2)lr(0.07)");

		LOG.info("Loading data from {}", args[0]);
		File file = new File(args[0]);
		IExecutorService executor = new NNExecutorService();

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("result.csv")))) {
			writer.write("ImageId,Label\r\n");

			LOG.info("Executing model");
			int[] predictions = Files.readLines(file, Charset.forName("utf-8")).stream()
					.map(String::trim)
					.filter(str -> !str.startsWith("pixel"))
					.map(str -> str.split(","))
					.map(CSVRunner::parseToDouble)
					.map(Utils::toINDArray)
					.mapToInt(features -> executor.getPredictionClass(model, features))
					.toArray();

			LOG.info("Writing predictions");
			for (int i = 0; i < predictions.length; ++i) {
				writer.write(Integer.toString(i + 1));
				writer.write(",");
				writer.write(Integer.toString(predictions[i]));
				writer.write("\r\n");
			}
		}
	}

	private static double[] parseToDouble(String[] arr) {
		return Stream.of(arr)
				.mapToDouble(Double::parseDouble)
				.toArray();
	}
}
