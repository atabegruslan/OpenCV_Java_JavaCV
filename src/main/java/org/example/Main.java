package org.example;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Main {
    private static Gender gender = new Gender();
    private static Age age = new Age();

    public static void main(String[] args) {
        Mat loadedImage = imread("./src/main/resources/oldman.PNG");

        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        cascadeClassifier.load("./src/main/resources/haarcascade_frontalface_alt.xml");

        RectVector detectObjects = new RectVector();
        cascadeClassifier.detectMultiScale(loadedImage, detectObjects);

        Map<Rect, Mat> detectedFaces = new HashMap<>();
        long numberOfPeople = detectObjects.size();
        for (int i = 0; i < numberOfPeople; i++) {
            Rect rect = detectObjects.get(i);
            Mat croppedMat = loadedImage.apply(new Rect(rect.x(), rect.y(), rect.width(), rect.height()));
            detectedFaces.put(rect, croppedMat);
        }

        detectedFaces.entrySet().forEach(rectMatEntry -> {
            String detectedGender = gender.detect(rectMatEntry.getValue());
            System.out.println(detectedGender);
            String detectedAge = age.detect(rectMatEntry.getValue());
            System.out.println(detectedAge);
        });

        //imwrite("./src/main/resources/destimg.PNG", loadedImage);
    }
}

class Gender {
    public String detect(Mat face) {
        try {
            Net genderNet = new Net();
            File protobuf = new File(getClass().getResource("/deploy_gendernet.prototxt").toURI());
            File caffeModel = new File(getClass().getResource("/gender_net.caffemodel").toURI());
            genderNet = readNetFromCaffe(protobuf.getAbsolutePath(), caffeModel.getAbsolutePath());

            Mat croppedMat = new Mat();
            resize(face, croppedMat, new Size(256, 256));
            //normalize(croppedMat, croppedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);
            Mat inputBlob = blobFromImage(croppedMat);
            genderNet.setInput(inputBlob, "data", 1.0, null);
            Mat prob = genderNet.forward("prob");
            Indexer indexer = prob.createIndexer();

            if (indexer.getDouble(0, 0) > indexer.getDouble(0, 1)) {
                return "Male";
            } else {
                return "Female";
            }
        } catch (Exception e) {
            throw new IllegalStateException("Unable to start Gender Detection", e);
        }
    }
}

class Age {
    private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-100"};

    public String detect(Mat face) {
        try {
            Net ageNet = new Net();
            File protobuf = new File(getClass().getResource("/deploy_agenet.prototxt").toURI());
            File caffeModel = new File(getClass().getResource("/age_net.caffemodel").toURI());
            ageNet = readNetFromCaffe(protobuf.getAbsolutePath(), caffeModel.getAbsolutePath());

            Mat resizedMat = new Mat();
            resize(face, resizedMat, new Size(256, 256));
            Mat inputBlob = blobFromImage(resizedMat);
            ageNet.setInput(inputBlob, "data", 1.0, null);      //set the network input
            Mat prob = ageNet.forward("prob");
            DoublePointer pointer = new DoublePointer(new double[1]);
            Point max = new Point();
            minMaxLoc(prob, null, pointer, null, max, null);

            return AGES[max.x()];
        } catch (Exception e) {
            throw new IllegalStateException("Unable to start Age Detection", e);
        }
    }
}

