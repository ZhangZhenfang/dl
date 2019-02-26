package peer.afang.dl.test;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Arctan;
import peer.afang.dl.util.Tanh;

/**
 * @author ZhangZhenfang
 * @date 2019/2/8 11:19
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
        Cnn cnn = new Cnn();
        Mat[] inputs = new Mat[4];
        Mat[] labels = new Mat[4];
        for (int i = 0; i < inputs.length; i++) {
            Mat m = new Mat(4, 4, CvType.CV_32F);
            m.put(0, 0, images[i]);
            Mat l = new Mat(1, 1, CvType.CV_32F);
            l.put(0, 0, Cnn.labels[i]);
            inputs[i] = m;
            labels[i] = l;
        }
        cnn.forward(inputs[0]);
        cnn.fcLayer.computeGrad(labels[0]);
        cnn.convLayer2.computeGrad();
        cnn.convLayer1.computeGrad();
        System.out.println(cnn.convLayer2.getGrad().dump());
        System.out.println(cnn.convLayer1.getGrad().dump());
    }

    private ConvLayer convLayer1;
    private ConvLayer convLayer2;
    private FCLayer fcLayer;
    public Cnn() {
        this.convLayer1 = new ConvLayer(2, new Tanh(), 4);
        Mat w = new Mat(2, 2, CvType.CV_32F);
        w.put(0, 0, data1);
        convLayer1.setWeight(w);
        this.convLayer2 = new ConvLayer(2, new Arctan(), 3);
        w = new Mat(2, 2, CvType.CV_32F);
        w.put(0, 0, data2);
        convLayer2.setWeight(w);
        convLayer1.setPreviousLayer(null);
        convLayer1.setNextLayer(convLayer2);
        convLayer2.setPreviousLayer(convLayer1);
        convLayer2.setNextLayer(null);
        convLayer2.setInput(convLayer1.getActivatedOut());
        this.fcLayer = new FCLayer(4, 1);
        this.fcLayer.setInput(convLayer2.getActivatedOut());
        w = new Mat(4, 1, CvType.CV_32F);
        w.put(0, 0, data3);
        fcLayer.setWeight(w);
        convLayer2.setFcLayer(fcLayer);
    }

    public void forward(Mat input) {
        convLayer1.setInput(input);
        convLayer1.computeOut();
        System.out.println(convLayer1.getActivatedOut().dump());
        convLayer2.computeOut();
        fcLayer.computeOut();
        System.out.println(fcLayer.getOut().dump());
    }
}
