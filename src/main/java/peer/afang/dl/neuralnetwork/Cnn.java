package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Arctan;
import peer.afang.dl.util.Tanh;

/**
 * @author ZhangZhenfang
 * @date 2019/2/3 18:47
 */
public class Cnn {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    static double[] x1 = new double[]{0, 0, 0, -1, -1, 0, -1, 0, -1, 0, -1, -1, 1, 0, -1, -1};
    static double[] x2 = new double[]{0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,1, 0, 0, -1};
    static double[] x3 = new double[]{0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 1, 1, 1, 0, -1, 1};
    static double[] x4 = new double[]{0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1};
    static double[][] labels = new double[][]{{-1.42889927219}, {-0.785398163397}, {0}, {1.46013910562}};
    static double[][] images = new double[][]{x1, x2, x3, x4};
    static double[] data1 = new double[]{-0.17385154, 2.80282212, -3.52931765, 2.07933109};
    static double[] data2 = new double[]{-5.08057429, -2.24798369, 1.37728729, 6.35670686};
    static double[] data3 = new double[]{4.13493441, -0.14675518, 0.02265047, 2.38585956};

    public static void main(String[] args) {
        Mat[] inputs = new Mat[4];
        for (int i = 0; i < inputs.length; i++) {
            Mat m = new Mat(4, 4, CvType.CV_32F);
            m.put(0, 0, images[i]);
            inputs[i] = m;
        }
        Cnn cnn = new Cnn();
        cnn.forward(inputs[0]);
    }

    NewConvLayer l1;

    NewConvLayer l2;

    Mat weight;
    Mat hid;

    public Cnn() {
        l1 = new NewConvLayer(2, 4, data1, new Tanh());
        l2 = new NewConvLayer(2, 3, data2, new Arctan());
        l2.setInput(l1.getActivatedOutput());
        weight = new Mat(data3.length, 1, CvType.CV_32F);
        weight.put(0, 0, data3);
        hid = new Mat();
    }

    public void forward(Mat input) {
        l1.setInput(input);
        l1.computeOut();
        l2.computeOut();
        Mat m = l2.getActivatedOutput();
        m = m.reshape(1, 1);
        Core.gemm(m, weight, 1, new Mat(), 1, hid);
    }

    public void back(Mat label) {
        Core.subtract();
    }
}
