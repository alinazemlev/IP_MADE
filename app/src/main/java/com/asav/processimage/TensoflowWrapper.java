package com.asav.processimage;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class TensoflowWrapper {

    private static final int INPUT_SIZE = 224;
    private static final String INPUT_NAME = "input_1";
    private static final String[] OUTPUT_NAMES = {"global_pooling/Mean","age_pred/Softmax","gender_pred/Sigmoid"};
    private static final String MODEL_FILE = "file:///android_asset/age_gender_tf2_new-01-0.14-0.92.pb";
    private TensorFlowImageFeatureExtractor featureExtractor=null;
    private Executor executor = Executors.newSingleThreadExecutor();
    public String res;

    public int getInputSize(){return INPUT_SIZE;};


    public void initTensorFlowAndLoadModel(AssetManager assetManager) {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    featureExtractor=new TensorFlowImageFeatureExtractor(
                            assetManager,
                            MODEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAMES);
                    //makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    public void classifyImage(Bitmap bmp) {
        float[][] cnn_outputs=featureExtractor.recognizeImage(bmp);
        float[] features=cnn_outputs[0];
        //age
        final float[] age_features=cnn_outputs[1];
        ArrayList<Integer> indices = new ArrayList<>();
        for (int j=0;j<age_features.length;++j){
            indices.add(j);
        }
        Collections.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer idx1, Integer idx2) {
                if (age_features[idx1]==age_features[idx2])
                    return 0;
                else if (age_features[idx1]>age_features[idx2])
                    return -1;
                else
                    return 1;
            }
        });
        int max_index=2;
        double[] probabs=new double[max_index];
        double sum=0;
        for(int j=0;j<max_index;++j){
            probabs[j]=age_features[indices.get(j)];
            sum+=probabs[j];
        }
        double age=0;
        for(int j=0;j<max_index;++j) {
            age+=(indices.get(j)+0.5)*probabs[j]/sum;
        }

        int a = (int) Math.round(age);
        float gender=cnn_outputs[2][0];
        String gen = gender>=0.6?" male":" female";
        res = "Gender: "+ gen + " Age: "+ String.valueOf(a);
    }

}
