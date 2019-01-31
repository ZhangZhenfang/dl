package peer.afang.dl;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import peer.afang.dl.neuralnetwork.cnn.Cnn;
import peer.afang.dl.neuralnetwork.cnn.ConvLayer;
import peer.afang.dl.neuralnetwork.cnn.PoolingLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * @author ZhangZhenfang
 * @date 2019/1/28 16:23
 */
public class Test {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }

    public static void main(String[] args) throws Exception {
        List<Mat> mats = new ArrayList<Mat>();
        Mat m = new Mat(1, 9, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        mats.add(m);
        m = new Mat(1, 9, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        mats.add(m);

        m = new Mat(4, 4, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        Mat dst = new Mat(m.size(), m.type());
        Core.flip(m, dst, 0);
        Core.flip(dst, dst, 1);
        System.out.println(dst.dump());

//        System.out.println(m.dump());
//        Mat dst = new Mat(2, 2, CvType.CV_32F);
//        Mat ker = Mat.ones(3, 3, CvType.CV_32F);
//        Imgproc.filter2D(m, dst, m.depth(), ker, new Point(-1, -1), 0, 3);

//        System.out.println(dst.dump());
//        Mat dst = new Mat(2, 2, CvType.CV_32F);
//        Mat posDst = new Mat(2, 2, CvType.CV_16U);
//        Cnn.maxPooling(m, 2, 2, dst, posDst);
//        System.out.println(m.dump());
//        System.out.println(dst.dump());
//        System.out.println(posDst.dump());
//
//        Mat upsample = PoolingLayer.upsample(dst, posDst);
//        System.out.println(upsample.dump());
        //        Core.vconcat(mats, dst);
//        System.out.println(dst.dump());
    }
}
