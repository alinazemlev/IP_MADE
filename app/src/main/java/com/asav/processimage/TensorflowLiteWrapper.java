package com.asav.processimage;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Pair;

import com.asav.processimage.classifiers.TFLiteImageClassifier;
import com.asav.processimage.utils.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public class TensorflowLiteWrapper {
    private TFLiteImageClassifier classif=null;
    private final String MODEL_FILE_NAME = "simple_classifier.tflite";
    private final int SCALED_IMAGE_BIGGEST_SIZE = 480;
    public ArrayList<Pair<String, String>> faceGroup= new ArrayList<>();
    public String res;


    public void initModel(AssetManager assetManager, String[] labels) {
        classif = new TFLiteImageClassifier(
                assetManager,
                MODEL_FILE_NAME,
               labels);
    }
    public void classifyEmotions(Bitmap imageBitmap) {
        Map<String, Float> result = classif.classify(imageBitmap, true);

        // Sort by increasing probability
        LinkedHashMap<String, Float> sortedResult =
                (LinkedHashMap<String, Float>) SortingHelper.sortByValues(result);

        ArrayList<String> reversedKeys = new ArrayList<>(sortedResult.keySet());
        // Change the order to get a decrease in probabilities
        Collections.reverse(reversedKeys);


        for (String key : reversedKeys) {
            res = "Emotion: "+ key;
            break;

        }

    }




}
