package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import peer.afang.dl.util.*;

/**
 * cnn算法简单实现，输入4*4，w1为2*2，w2为2*2，全连接w为4*2，输出1*2
 * @author ZhangZhenfang
 * @date 2019/2/26 18:58
 */
public class TestCnn {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }

    static double[] x1 = new double[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0};
    static double[] x2 = new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0};
    static double[] x3 = new double[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0};
    static double[] x4 = new double[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0};
    static double[][] labels = new double[][]{{0, 1}, {0, 1}, {1, 0}, {1, 0}};
    static double[][] images = new double[][]{x1, x2, x3, x4};
    static double[] data1 = new double[]{-0.17385154, 0.80282212, -0.52931765, 0.07933109};
    static double[] data2 = new double[]{0.08057429, -0.24798369, 0.37728729, -0.35670686};
    static double[] data3 = new double[]{-0.13493441, -0.14675518, 0.02265047, 0.38585956};

    public static void main(String[] args) {
        Mat[] inputs = new Mat[4];
        Mat[] labels = new Mat[4];
        Mat[] weights = new Mat[3];

        for (int i = 0; i < inputs.length; i++) {
            Mat m = new Mat(4, 4, CvType.CV_32F);
            m.put(0, 0, images[i]);
            Mat l = new Mat(1, 2, CvType.CV_32F);
            l.put(0, 0, TestCnn.labels[i]);
            inputs[i] = m;
            System.out.println(inputs[i].dump());
            labels[i] = l;
            System.out.println(labels[i].dump());
        }
        weights[0] = new Mat(2, 2, CvType.CV_32F);
        weights[0].put(0, 0, data1);
        weights[1] = new Mat(2, 2, CvType.CV_32F);
        weights[1].put(0, 0, data2);
        weights[2] = new Mat(4, 2, CvType.CV_32F);
        weights[2].put(0, 0, data3);
        double b1 = 0, b2 = 0, b3 = 0;
        Activator relu = new Relu();
        Activator sigmoid = new Sigmoid();

        for (int j = 0; j < 5000; j++) {
            for (int i = 0; i < 4; i++) {
                // 第一层卷积层
                Mat z1 = MatUtils.conv(inputs[i], weights[0], 1, 0, 0);
                Core.add(z1, new Scalar(b1), z1);
                Mat a1 = relu.activate(z1);

                // 第二层卷积层
                Mat z2 = MatUtils.conv(a1, weights[1], 1, 0, 0);
                Core.add(z2, new Scalar(b2), z2);
                Mat a2 = relu.activate(z2);

                // 全连接层
                Mat z3 = new Mat();
                Core.gemm(a2.reshape(1, 1), weights[2], 1, new Mat(), 1, z3);
                Mat t1 = new Mat(1, 1, CvType.CV_32F);
                t1.put(0, 0, b3);
                Core.add(z3, new Scalar(b3), z3);
                Mat a3 = sigmoid.activate(z3);

                // 计算全连接层delta
                Mat delta3 = new Mat();
                Mat derivativeZ3 = sigmoid.derivative(a3);
                Mat deltaOut = new Mat();
                Core.subtract(labels[i], a3, deltaOut);
                Core.multiply(deltaOut, derivativeZ3, delta3);
                // 计算全连接层梯度
                Mat grad3 = new Mat();
                Core.gemm(a2.reshape(1, 1).t(), delta3, 1, new Mat(), 1, grad3);

                // 计算第二层卷积层delta
                Mat delta2 = new Mat();
                Mat derivativeZ2 = relu.derivative(a2);
                Mat t2 = new Mat();
                Core.gemm(weights[2], delta3.t(), 1, new Mat(), 1, t2);
                Mat t3 = t2.reshape(1, 2);
                Core.multiply(t3, derivativeZ2, delta2);
                // 计算第二层卷积层梯度
                Mat grad2 = MatUtils.conv(a1, delta2, 1, 0, 0);

                // 计算第一层卷积层delta
                Mat delta1 = new Mat();
                Mat paddedDelta2 = MatUtils.paddingZeor(delta2, 1, 1, 1, 1);
                Mat conv = MatUtils.conv(paddedDelta2, MatUtils.rotate180(weights[1]), 1, 0, 0);
                Mat derivativeZ1 = relu.derivative(a1);
                Core.multiply(conv, derivativeZ1, delta1);
                // 计算第一层卷积层梯度
                Mat grad1 = MatUtils.conv(inputs[i], delta1, 1, 0, 0);


                /**
                 * 更新权重和偏执
                 */
                Mat mm = new Mat();
                Core.multiply(grad3, new Scalar(0.1), mm);
                Core.add(weights[2], mm, weights[2]);
                b3 = b3 - 0.1 * MatUtils.sumMat(delta3);
                Core.multiply(grad2, new Scalar(0.1), mm);
                Core.add(weights[1], mm, weights[1]);
                b2 = b2 - 0.1 * MatUtils.sumMat(delta2);
                Core.multiply(grad1, new Scalar(0.1), mm);
                Core.add(weights[0], mm, weights[0]);
                b1 = b1 - 0.1 * MatUtils.sumMat(delta1);
            }
        }

        System.out.println("weights");
        System.out.println(weights[0].dump());
        System.out.println(b1);
        System.out.println(weights[1].dump());
        System.out.println(b2);
        System.out.println(weights[2].dump());
        System.out.println(b3);
        System.out.println("\n\n");

        // 测试
        for (int i = 0; i < 4; i++) {
            Mat z1 = MatUtils.conv(inputs[i], weights[0], 1, 0, 0);
            Core.add(z1, new Scalar(b1), z1);
            Mat a1 = relu.activate(z1);

            Mat z2 = MatUtils.conv(a1, weights[1], 1, 0, 0);
            Core.add(z2, new Scalar(b2), z2);
            Mat a2 = relu.activate(z2);

            Mat z3 = new Mat();
            Core.gemm(a2.reshape(1, 1), weights[2], 1, new Mat(), 1, z3);
            Core.add(z3, new Scalar(b3), z3);
            Mat a3 = sigmoid.activate(z3);
            System.out.println("fc");
            System.out.println(z3.dump());
            System.out.println(a3.dump());
        }
    }
}
