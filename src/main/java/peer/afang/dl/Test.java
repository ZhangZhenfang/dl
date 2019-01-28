package peer.afang.dl;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

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

    public static void main(String[] args) {
        List<Mat> mats = new ArrayList<Mat>();
        Mat m = new Mat(1, 9, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        mats.add(m);
        m = new Mat(1, 9, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        mats.add(m);

        m = new Mat(1, 9, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        mats.add(m);
        Mat dst = new Mat(3, 9, CvType.CV_32F);
        Core.vconcat(mats, dst);
        System.out.println(dst.dump());
    }
}
