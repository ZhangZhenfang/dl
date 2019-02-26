package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import peer.afang.dl.util.Arctan;
import peer.afang.dl.util.MnistReader;
import peer.afang.dl.util.MnistTest;
import peer.afang.dl.util.Tanh;

import java.util.Random;
import java.util.Scanner;

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

    public static double findIndex(double[] data) {
        for (int i= 0; i < data.length; i++) {
            if (data[i] == 1) {
                return i + 1;
            }
        }
        return -1;
    }
    public static void main(String[] args) {
        Cnn cnn = new Cnn();
        System.out.println(cnn.l1.getWeight().dump());
        MnistReader mnistReader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        int count = 0;
        while (true) {
            double[] nextImage = mnistReader.getNextImage(true);
            if (nextImage == null) {
               break;
            }
            Mat m = new Mat(28, 28, CvType.CV_32F);
            m.put(0, 0, nextImage);
            cnn.forward(m);
//            MnistTest.getNextLabel(mnistReader);
//            double[] nextLabel = mnistReader.getNextLabel();
            double l = findIndex(nextImage);
            Mat label = new Mat(1, 1, CvType.CV_32F);
            label.put(0, 0, l);
            cnn.back(label);
            if (count++ % 1000 == 0) {
                System.out.println(cnn.l1.getWeight().dump());
                System.out.println(count);
                mnistReader.open(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
                Mat in = new Mat();
                double[] nextLabel = mnistReader.getNextLabel();
                in.put(0, 0, nextLabel);
                cnn.forward(in);
                System.out.println(cnn.activatedOut.dump());
            }
        }


    }

    public static void test() {
        Mat[] inputs = new Mat[4];
        for (int i = 0; i < inputs.length; i++) {
            Mat m = new Mat(4, 4, CvType.CV_32F);
            m.put(0, 0, images[i]);
            inputs[i] = m;
        }
        Cnn cnn = new Cnn();
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 4; j++) {
                cnn.forward(inputs[j]);
                Mat label = new Mat(1, 1, CvType.CV_32F);
                label.put(0, 0, labels[j]);
                cnn.back(label);
            }
        }

        cnn.forward(inputs[0]);
        System.out.println(cnn.activatedOut.dump());
        cnn.forward(inputs[1]);
        System.out.println(cnn.activatedOut.dump());
        cnn.forward(inputs[2]);
        System.out.println(cnn.activatedOut.dump());
        cnn.forward(inputs[3]);
        System.out.println(cnn.activatedOut.dump());
    }
    NewConvLayer l1;

    NewConvLayer l2;
    NewConvLayer l3;

    Mat weight;
    Mat hid;
    Mat out;
    Mat activatedOut;

    public Cnn() {
        l1 = new NewConvLayer(6, 28, new Tanh(), false);
//        System.out.println(l1.getOut());
        l2 = new NewConvLayer(6, 23, new Tanh(), false);
//        System.out.println(l2.getOut());
        l3 = new NewConvLayer(6, 18, new Arctan(), true);
//        System.out.println(l3.getOut());
        l2.setInput(l1.getActivatedOutput());
        l3.setInput(l2.getActivatedOutput());
        weight = new Mat(l3.getActivatedOutput().rows() * l3.getActivatedOutput().cols(), 1, CvType.CV_32F);
        double[] data = new double[weight.rows() * weight.cols()];
        int index = 0;
        for (int i = 0; i < data.length; i++) {
            data[index++] = (new Random().nextDouble() - 0.5) * 2;
        }
        weight.put(0, 0, data);
        hid = new Mat();
        out = new Mat();
        activatedOut = new Mat();
    }

    public void forward(Mat input) {
        l1.setInput(input);
        l1.computeOut();
//        System.out.println(l1.getActivatedOutput().dump());
//        new Scanner(System.in).nextLine();
        l2.computeOut();
        l3.computeOut();
        Mat m = l3.getActivatedOutput();
        hid = m.reshape(1, 1);
        Core.gemm(hid, weight, 1, new Mat(), 1, out);
        activatedOut = new Arctan().activate(out);
    }

    public void back(Mat label) {
        Mat part1 = new Mat();
        Core.subtract(activatedOut, label, part1);

        Mat part2 = new Arctan().derivative(out);
        Mat part3 = hid;
        Mat dst = new Mat();
        Core.multiply(part1, part2, dst);
        Mat grad3 = new Mat();
        Core.gemm(part3.t(), dst, 1, new Mat(), 1, grad3);

        l3.computeGrad(part1, part2, weight);
        l2.computeGrad(l3.getPart1(), l3.getPart2(), l3.getWeight());
        l1.computeGrad(l2.getPart1(), l2.getPart2(), l2.getWeight());

        Mat t = new Mat();
        Core.multiply(grad3, new Scalar(0.1), t);
        Core.subtract(weight, t, weight);
        l3.updateWeight(0.1);
        l2.updateWeight(0.1);
        l1.updateWeight(0.1);
    }
}
