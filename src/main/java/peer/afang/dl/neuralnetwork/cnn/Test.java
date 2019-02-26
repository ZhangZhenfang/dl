package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Activator;
import peer.afang.dl.util.MatUtils;
import peer.afang.dl.util.Relu;
import peer.afang.dl.util.Sigmoid;

/**
 * @author ZhangZhenfang
 * @date 2019/2/26 18:58
 */
public class Test {
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
    static double[] data1 = new double[]{-0.17385154, 0.80282212, -0.52931765, 0.07933109};
    static double[] data2 = new double[]{-0.08057429, -0.24798369, 0.37728729, 0.35670686};
    static double[] data3 = new double[]{0.13493441, -0.14675518, 0.02265047, 0.38585956};

    public static void main(String[] args) {
        Mat[] inputs = new Mat[4];
        Mat[] labels = new Mat[4];
        Mat[] weights = new Mat[3];
        for (int i = 0; i < inputs.length; i++) {
            Mat m = new Mat(4, 4, CvType.CV_32F);
            m.put(0, 0, images[i]);
            Mat l = new Mat(1, 1, CvType.CV_32F);
            l.put(0, 0, Test.labels[i]);
            inputs[i] = m;
            System.out.println(inputs[i].dump());
            labels[i] = l;
            System.out.println(labels[i].dump());
        }
        weights[0] = new Mat(2, 2, CvType.CV_32F);
        weights[0].put(0, 0, data1);
        weights[1] = new Mat(2, 2, CvType.CV_32F);
        weights[1].put(0, 0, data2);
        weights[2] = new Mat(4, 1, CvType.CV_32F);
        weights[2].put(0, 0, data3);
        double b1 = 0, b2 = 0, b3 = 0;
        Activator relu = new Relu();
        Activator sigmoid = new Sigmoid();
        Mat z1 = MatUtils.conv(inputs[0], weights[0], 1, 0, b1);
        Mat a1 = relu.activate(z1);
        System.out.println("conv1");
        System.out.println(z1.dump());
        System.out.println(a1.dump());

        Mat z2 = MatUtils.conv(a1, weights[1], 1, 0, b2);
        Mat a2 = relu.activate(z2);
        System.out.println("conv2");
        System.out.println(z2.dump());
        System.out.println(a2.dump());

        Mat z3 = new Mat();
        Core.gemm(a2.reshape(1, 1), weights[2], 1, new Mat(), 1, z3);
        Mat t1 = new Mat(1, 1, CvType.CV_32F);
        t1.put(0, 0, b3);
        Core.add(z3, t1, z3);
        Mat a3 = sigmoid.activate(z3);
        System.out.println("fc");
        System.out.println(z3.dump());
        System.out.println(a3.dump());

        Mat delta3 = new Mat();
        Mat derivativeZ3 = sigmoid.derivative(z3);
        System.out.println(derivativeZ3.dump());
        Mat deltaOut = new Mat();
        Core.subtract(labels[0], a3, deltaOut);
        Core.multiply(deltaOut, derivativeZ3, delta3);
        System.out.println("delta3");
        System.out.println(delta3.dump());
        Mat grad3 = new Mat();
        Core.gemm(a2.reshape(1, 1).t(), delta3, 1, new Mat(), 1, grad3);
        System.out.println(grad3.dump());


        System.out.println("delta2");
        Mat delta2 = new Mat();
        Mat derivativeZ2 = relu.derivative(z2);
        Mat t2 = new Mat();
        Core.gemm(weights[2], delta3, 1, new Mat(), 1, t2);
        Mat t3 = t2.reshape(1, 2);
        Core.multiply(t3, derivativeZ2, delta2);
        System.out.println(delta2.dump());

        Mat grad2 = MatUtils.conv(a1, delta2, 1, 0, 0);
        System.out.println(grad2.dump());

        System.out.println("delta1");
        Mat delta1 = new Mat();
        Mat derivativeZ1 = relu.derivative(z1);
        Mat t4 = new Mat();
        Core.gemm(weights[2], delta3, 1, new Mat(), 1, t2);
        Mat t5 = t4.reshape(1, 2);
        Core.multiply(t5, derivativeZ1, delta1);
        System.out.println(delta1.dump());

        Mat grad1 = MatUtils.conv(inputs[0], delta1, 1, 0, 0);
        System.out.println(grad1.dump());

    }
}
