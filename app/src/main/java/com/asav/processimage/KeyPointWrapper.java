package com.asav.processimage;
import android.content.Context;
import android.graphics.Bitmap;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class KeyPointWrapper {
    private File mCascadeFile;
    private File mCascadeFileEY;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mCascadeEYE;
    private float mRelativeSize = 0.1f;
    private Mat mRgba;
    private Mat mGray;
    String FILENAME ="banana4th.xml";
    CascadeClassifier detector = null;

    public KeyPointWrapper(Context context) throws IOException {
        init(context);
    }

    public void init(Context context) throws IOException {
//        InputStream is = getResources().openRawResource(
//                R.raw.haarcascade_frontalface_default);
//        File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
//        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
//        FileOutputStream os = new FileOutputStream(mCascadeFile);
//
//        byte[] buffer = new byte[4096];
//        int bytesRead;
//        while ((bytesRead = is.read(buffer)) != -1) {
//            os.write(buffer, 0, bytesRead);
//        }
//        is.close();
//        os.close();

        // ------------------------- load left eye classificator//
        // -----------------------------------
        InputStream iser = context.getResources().openRawResource(
                R.raw.haarcascade_eye);
        File cascadeDirEy = context.getDir("cascadeER",
                Context.MODE_PRIVATE);
        mCascadeFileEY = new File(cascadeDirEy,
                "haarcascade_eye_right.xml");
        FileOutputStream oser = new FileOutputStream(mCascadeFileEY);

        byte[] bufferER = new byte[4096];
        int bytesReadER;
        while ((bytesReadER = iser.read(bufferER)) != -1) {
            oser.write(bufferER, 0, bytesReadER);
        }
        iser.close();
        oser.close();

        mJavaDetector = new CascadeClassifier(
                mCascadeFile.getAbsolutePath());
        mCascadeEYE = new CascadeClassifier(cascadeDirEy.getAbsolutePath());

        cascadeDirEy.delete();

    }
    public Mat predict(Mat rgba, Mat gray){

        int mAbsoluteSize = 0;
        MatOfRect lands = new MatOfRect();
        int height = gray.rows();

        mAbsoluteSize = Math.round(height * mRelativeSize);
        mCascadeEYE.detectMultiScale(gray, lands, 1.3, 5, 2, //TODO: objdetect.CV_HAAR_SCALE_IMAGE)
                new Size(mAbsoluteSize, mAbsoluteSize), new Size());

        Rect[] EyeArray = lands.toArray();
        for (int i = 0; i < EyeArray.length; i++) {
            Imgproc.rectangle(rgba, EyeArray[i].tl(),
                    EyeArray[i].br(), new Scalar(0, 255, 0), 3);
            double xCenter = (EyeArray[i].x + EyeArray[i].width + EyeArray[i].x) / 2;
            double yCenter = (EyeArray[i].y + EyeArray[i].height + EyeArray[i].y) / 2;
        }
        return rgba;

    }

}



