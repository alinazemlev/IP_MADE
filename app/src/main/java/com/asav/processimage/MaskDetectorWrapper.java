package com.asav.processimage;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.asav.processimage.classifiers.InterpreterImageParams;
import com.asav.processimage.classifiers.TFLiteClassifier;
import com.asav.processimage.classifiers.behaviors.ClassifyBehavior;
import com.asav.processimage.classifiers.behaviors.TFLiteImageClassification;
import com.asav.processimage.utils.SortingHelper;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.label.TensorLabel;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class MaskDetectorWrapper extends TFLiteClassifier {
    TensorImage inputImageBuffer;
    String labs;

    public MaskDetectorWrapper(AssetManager assetManager, String modelFileName, String[] labels) {
        super(assetManager, modelFileName, labels);


    }



    public void predict(Bitmap input){
        // preprocess
        int mImageWidth = InterpreterImageParams.getInputImageWidth(mInterpreter);
        int mImageHeight = InterpreterImageParams.getInputImageHeight(mInterpreter);
        inputImageBuffer = new TensorImage(mInterpreter.getInputTensor(0).dataType());


        int cropSize = Math.min(input.getWidth(), input.getHeight());
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(mImageWidth, mImageWidth, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new NormalizeOp(127.5f, 127.5f))
                        .build();


        // load image
        inputImageBuffer.load(input);
        inputImageBuffer = imageProcessor.process(inputImageBuffer);
        DataType outputDataType = mInterpreter.getOutputTensor(0).dataType();
        int[] outputShape = mInterpreter.getOutputTensor(0).shape();
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType);


        // run model
        mInterpreter.run(inputImageBuffer.getBuffer(), outputBuffer.getBuffer().rewind());

        // get output
        TensorLabel labelOutput = new TensorLabel(mLabels, outputBuffer);

        Map<String, Float> label = labelOutput.getMapWithFloatValue();
        LinkedHashMap<String, Float> sortedResult =
                (LinkedHashMap<String, Float>) SortingHelper.sortByValues(label);

        ArrayList<String> reversedKeys = new ArrayList<>(sortedResult.keySet());
        // Change the order to get a decrease in probabilities
        Collections.reverse(reversedKeys);


        for (String key : reversedKeys) {
            labs = " "+ key;
            break;

        }



    }

}
